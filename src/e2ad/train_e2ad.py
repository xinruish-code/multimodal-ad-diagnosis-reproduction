import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from e2ad.data import E2ADDataset, collate, read_manifest
from e2ad.model import (
    E2ADStudent,
    E2ADTeacher,
    RelationProjector,
    anatomy_kd_loss,
    logit_kd_loss,
    mapper_regulation_loss,
    relation_kd_loss,
)


def parse_shape(value):
    return tuple(int(v) for v in value.split(","))


def metrics(y_true, prob):
    pred = prob.argmax(axis=1)
    result = {"acc": accuracy_score(y_true, pred), "bca": balanced_accuracy_score(y_true, pred)}
    if prob.shape[1] == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        result.update({
            "sen": tp / max(tp + fn, 1),
            "spec": tn / max(tn + fp, 1),
            "pre": precision_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "auc": roc_auc_score(y_true, prob[:, 1]) if len(np.unique(y_true)) == 2 else float("nan"),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        })
        if len(np.unique(y_true)) == 2:
            best = {"thr_f1": -1.0}
            for thr in np.linspace(0.05, 0.95, 181):
                thr_pred = (prob[:, 1] >= thr).astype(int)
                ttn, tfp, tfn, ttp = confusion_matrix(y_true, thr_pred, labels=[0, 1]).ravel()
                thr_f1 = f1_score(y_true, thr_pred, zero_division=0)
                thr_bca = balanced_accuracy_score(y_true, thr_pred)
                if (thr_f1, thr_bca) > (best["thr_f1"], best.get("thr_bca", -1.0)):
                    best = {
                        "best_thr": float(thr),
                        "thr_acc": accuracy_score(y_true, thr_pred),
                        "thr_bca": thr_bca,
                        "thr_sen": ttp / max(ttp + tfn, 1),
                        "thr_spec": ttn / max(ttn + tfp, 1),
                        "thr_pre": precision_score(y_true, thr_pred, zero_division=0),
                        "thr_f1": thr_f1,
                        "thr_tn": int(ttn),
                        "thr_fp": int(tfp),
                        "thr_fn": int(tfn),
                        "thr_tp": int(ttp),
                    }
            result.update(best)
    else:
        result["pre"] = precision_score(y_true, pred, average="macro", zero_division=0)
        result["f1"] = f1_score(y_true, pred, average="macro", zero_division=0)
        result["auc"] = roc_auc_score(y_true, prob, multi_class="ovr") if len(np.unique(y_true)) > 2 else float("nan")
    return result


def class_weights_from_targets(targets, num_classes, device):
    counts = np.bincount(np.asarray(targets), minlength=num_classes).astype("float32")
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def metric_score(stats, metric_name):
    value = stats.get(metric_name)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return stats.get("auc", stats.get("acc", -1.0))
    return value


def run_teacher_epoch(teacher, loader, optimizer, device, args, class_weight=None, collect_outputs=False, debug_shapes=False):
    train = optimizer is not None
    teacher.train(train)
    total = 0.0
    all_y, all_prob, all_id, all_feat, all_weights = [], [], [], [], []
    debug_once = debug_shapes
    iterator = tqdm(loader, leave=False, desc="teacher" if train else "teacher_eval")
    for batch in iterator:
        mri = batch["mri"].to(device)
        pet = batch["pet"].to(device)
        target = batch["target"].to(device)
        with torch.set_grad_enabled(train):
            out = teacher(mri, pet, debug_shapes=debug_once)
            debug_once = False
            loss = F.cross_entropy(out["logits"], target, weight=class_weight) + args.reg_weight * mapper_regulation_loss(out)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        prob = torch.softmax(out["logits"], dim=1).detach().cpu().numpy()
        all_prob.append(prob)
        all_y.append(target.detach().cpu().numpy())
        if collect_outputs:
            all_feat.append(out["feature"].detach().cpu().numpy())
            all_weights.append(out["weights"].detach().cpu().numpy())
            all_id.extend(batch["id"])
        total += float(loss.item()) * mri.size(0)
        iterator.set_postfix(loss=float(loss.item()))
    y = np.concatenate(all_y)
    prob = np.concatenate(all_prob)
    result = metrics(y, prob)
    result["loss"] = total / len(loader.dataset)
    outputs = None
    if collect_outputs:
        feat = np.concatenate(all_feat, axis=0)
        weights = np.concatenate(all_weights, axis=0)
        outputs = pd.DataFrame({"id": all_id, "y_true": y.astype(int)})
        for cls_idx in range(prob.shape[1]):
            outputs[f"prob_{cls_idx}"] = prob[:, cls_idx]
        if prob.shape[1] == 2:
            outputs["prob_positive"] = prob[:, 1]
        outputs = pd.concat([
            outputs,
            pd.DataFrame(feat, columns=[f"feat_{i:03d}" for i in range(feat.shape[1])]),
            pd.DataFrame(weights, columns=[f"roi_weight_{i:03d}" for i in range(weights.shape[1])]),
        ], axis=1)
    return result, outputs


def run_student_epoch(teacher, student, projector, loader, optimizer, device, args, class_weight=None, collect_outputs=False, debug_shapes=False):
    train = optimizer is not None
    teacher.eval()
    student.train(train)
    projector.train(train)
    total = 0.0
    all_y, all_prob, all_id, all_feat, all_weights = [], [], [], [], []
    debug_once = debug_shapes
    iterator = tqdm(loader, leave=False, desc="student" if train else "eval")
    for batch in iterator:
        mri = batch["mri"].to(device)
        pet = batch["pet"].to(device)
        target = batch["target"].to(device)
        with torch.no_grad():
            teacher_out = teacher(mri, pet)
        with torch.set_grad_enabled(train):
            student_out = student(mri, debug_shapes=debug_once)
            debug_once = False
            ce = F.cross_entropy(student_out["logits"], target, weight=class_weight)
            reg = mapper_regulation_loss(student_out)
            logit_kd = logit_kd_loss(teacher_out["logits"], student_out["logits"], args.kd_temperature)
            ana_kd = anatomy_kd_loss(teacher_out["weights"], student_out["weights"])
            rel_kd = relation_kd_loss(teacher_out, student_out, projector)
            loss = ce + args.reg_weight * reg + args.lambda_logit * logit_kd + args.lambda_ana * ana_kd + args.lambda_rel * rel_kd
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        prob = torch.softmax(student_out["logits"], dim=1).detach().cpu().numpy()
        all_prob.append(prob)
        all_y.append(target.detach().cpu().numpy())
        if collect_outputs:
            all_feat.append(student_out["feature"].detach().cpu().numpy())
            all_weights.append(student_out["weights"].detach().cpu().numpy())
            all_id.extend(batch["id"])
        total += float(loss.item()) * mri.size(0)
        iterator.set_postfix(loss=float(loss.item()))
    y = np.concatenate(all_y)
    prob = np.concatenate(all_prob)
    result = metrics(y, prob)
    result["loss"] = total / len(loader.dataset)
    outputs = None
    if collect_outputs:
        feat = np.concatenate(all_feat, axis=0)
        weights = np.concatenate(all_weights, axis=0)
        outputs = pd.DataFrame({"id": all_id, "y_true": y.astype(int)})
        for cls_idx in range(prob.shape[1]):
            outputs[f"prob_{cls_idx}"] = prob[:, cls_idx]
        if prob.shape[1] == 2:
            outputs["prob_positive"] = prob[:, 1]
        outputs = pd.concat([
            outputs,
            pd.DataFrame(feat, columns=[f"feat_{i:03d}" for i in range(feat.shape[1])]),
            pd.DataFrame(weights, columns=[f"roi_weight_{i:03d}" for i in range(weights.shape[1])]),
        ], axis=1)
    return result, outputs


def train_fold(args, frame, train_idx, val_idx, fold, label_to_idx):
    train_ds = E2ADDataset(frame.iloc[train_idx], args.input_shape, augment=True)
    val_ds = E2ADDataset(frame.iloc[val_idx], args.input_shape, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    teacher = E2ADTeacher(len(label_to_idx), args.roi_grid, args.base_channels, args.feature_dim, args.heads).to(device)
    student = E2ADStudent(len(label_to_idx), args.roi_grid, args.base_channels, args.feature_dim, args.heads).to(device)
    projector = RelationProjector(args.feature_dim).to(device)
    opt_teacher = torch.optim.AdamW(teacher.parameters(), lr=args.teacher_lr, weight_decay=args.weight_decay)
    opt_student = torch.optim.AdamW(list(student.parameters()) + list(projector.parameters()), lr=args.student_lr, weight_decay=args.weight_decay)
    out_dir = Path(args.output_dir) / args.task / f"fold_{fold:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    class_weight = class_weights_from_targets(frame.iloc[train_idx]["target"].to_numpy(), len(label_to_idx), device)

    best_teacher = {args.teacher_select_metric: -1.0}
    best_teacher_state = None
    for epoch in range(1, args.teacher_epochs + 1):
        train_stats, _ = run_teacher_epoch(teacher, train_loader, opt_teacher, device, args, class_weight, debug_shapes=args.debug_shapes and epoch == 1)
        val_stats, teacher_outputs = run_teacher_epoch(teacher, val_loader, None, device, args, class_weight, collect_outputs=True)
        print(json.dumps({"phase": "teacher_pretrain", "epoch": epoch, "train": train_stats, "val": val_stats}, ensure_ascii=False))
        if metric_score(val_stats, args.teacher_select_metric) > metric_score(best_teacher, args.teacher_select_metric):
            best_teacher = val_stats
            best_teacher_state = copy.deepcopy(teacher.state_dict())
            torch.save({"teacher": best_teacher_state, "args": vars(args), "labels": label_to_idx, "metrics": best_teacher}, out_dir / "best_teacher.pt")
            teacher_outputs.to_csv(out_dir / "teacher_val_predictions.csv", index=False)
    if best_teacher_state is not None:
        teacher.load_state_dict(best_teacher_state)

    best = {args.student_select_metric: -1.0}
    for epoch in range(1, args.student_epochs + 1):
        train_stats, _ = run_student_epoch(teacher, student, projector, train_loader, opt_student, device, args, class_weight, debug_shapes=args.debug_shapes and epoch == 1)
        val_stats, val_outputs = run_student_epoch(teacher, student, projector, val_loader, None, device, args, class_weight, collect_outputs=True)
        print(json.dumps({"phase": "student_distill", "epoch": epoch, "train": train_stats, "val": val_stats}, ensure_ascii=False))
        score = metric_score(val_stats, args.student_select_metric)
        if score > metric_score(best, args.student_select_metric):
            best = val_stats
            torch.save({"teacher": teacher.state_dict(), "student": student.state_dict(), "projector": projector.state_dict(), "args": vars(args), "labels": label_to_idx, "metrics": best}, out_dir / "best.pt")
            val_outputs.to_csv(out_dir / "val_predictions.csv", index=False)
    fold_metrics = {"student": best, "teacher": best_teacher}
    (out_dir / "metrics.json").write_text(json.dumps(fold_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return fold_metrics


def summarize_results(results):
    if isinstance(results[0], dict) and all(isinstance(v, dict) for v in results[0].values()):
        return {section: summarize_results([result[section] for result in results]) for section in results[0]}
    summary = {k: [float(r[k]) for r in results if k in r] for k in results[0]}
    count_keys = {"tn", "fp", "fn", "tp", "thr_tn", "thr_fp", "thr_fn", "thr_tp"}
    return {
        k: ({"sum": int(np.sum(v))} if k in count_keys else {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))})
        for k, v in summary.items()
    }


def main():
    parser = argparse.ArgumentParser(description="Reproduce E2AD with local paired MRI/PET ADNI-style data.")
    parser.add_argument("--data-root", default=r"D:\AD\心睿\ADNI-amyloid-smri-pet")
    parser.add_argument("--mri-folder", default="mwp1")
    parser.add_argument("--pet-folder", default="pet")
    parser.add_argument("--task", default="AD_CN", choices=["AD_CN", "MCI_CN", "AD_MCI", "CN_MCI_AD"])
    parser.add_argument("--output-dir", default="e2ad_runs")
    parser.add_argument("--input-shape", type=parse_shape, default=(64, 80, 64))
    parser.add_argument("--roi-grid", type=parse_shape, default=(4, 4, 4))
    parser.add_argument("--teacher-epochs", type=int, default=10)
    parser.add_argument("--student-epochs", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--single-split", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--reg-weight", type=float, default=0.01)
    parser.add_argument("--lambda-logit", type=float, default=1.0)
    parser.add_argument("--lambda-ana", type=float, default=1.0)
    parser.add_argument("--lambda-rel", type=float, default=1.0)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--teacher-select-metric", default="thr_f1")
    parser.add_argument("--student-select-metric", default="thr_f1")
    parser.add_argument("--teacher-lr", type=float, default=2e-4)
    parser.add_argument("--student-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-shapes", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    frame, label_to_idx = read_manifest(args.data_root, args.task, args.mri_folder, args.pet_folder)
    if args.max_samples is not None and args.max_samples < len(frame):
        per_class = max(1, args.max_samples // frame["target"].nunique())
        pieces = [group.sample(min(len(group), per_class), random_state=args.seed) for _, group in frame.groupby("target")]
        frame = pd.concat(pieces).reset_index(drop=True)
    print(f"Loaded {len(frame)} samples for {args.task}: {label_to_idx}")
    print(frame["Label"].value_counts().to_string())
    y = frame["target"].to_numpy()
    if args.single_split:
        train_idx, val_idx = train_test_split(np.arange(len(frame)), test_size=0.2, stratify=y, random_state=args.seed)
        results = [train_fold(args, frame, train_idx, val_idx, 0, label_to_idx)]
    else:
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        results = [train_fold(args, frame, train_idx, val_idx, fold, label_to_idx) for fold, (train_idx, val_idx) in enumerate(splitter.split(frame, y))]
    summary = summarize_results(results)
    out_dir = Path(args.output_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
