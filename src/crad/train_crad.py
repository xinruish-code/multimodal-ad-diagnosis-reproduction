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

from crad.data import CRADDataset, collate, read_manifest
from crad.model import CRADStudent, CRADTeacher, mmse_loss, orthogonal_loss, pf_cmad_loss, relation_kd_loss, smooth_confidence, soft_label_kd_loss


def parse_shape(value):
    return tuple(int(v) for v in value.split(","))


def parse_channels(value):
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
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })
    else:
        result["sen"] = result["bca"]
        result["spec"] = result["bca"]
        result["pre"] = precision_score(y_true, pred, average="weighted", zero_division=0)
        result["f1"] = f1_score(y_true, pred, average="weighted", zero_division=0)
        result["auc"] = roc_auc_score(y_true, prob, multi_class="ovr") if len(np.unique(y_true)) > 2 else float("nan")
    return result


def class_weights(targets, num_classes, device):
    counts = np.bincount(np.asarray(targets), minlength=num_classes).astype("float32")
    w = counts.sum() / np.maximum(counts, 1.0)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


def run_teacher(teacher, loader, optimizer, device, args, weight=None, collect=False, debug_shapes=False):
    train = optimizer is not None
    teacher.train(train)
    total = 0.0
    all_y, all_prob, all_id, all_feat = [], [], [], []
    debug_once = debug_shapes
    for batch in tqdm(loader, leave=False, desc="teacher" if train else "teacher_eval"):
        mri, pet, target, mmse = batch["mri"].to(device), batch["pet"].to(device), batch["target"].to(device), batch["mmse"].to(device)
        with torch.set_grad_enabled(train):
            out = teacher(mri, pet, debug_shapes=debug_once)
            debug_once = False
            loss = F.cross_entropy(out["logits"], target, weight=weight) + args.gamma_mmse * mmse_loss(out, mmse) + args.gamma_orth * orthogonal_loss(out)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        prob = torch.softmax(out["logits"], dim=1).detach().cpu().numpy()
        all_prob.append(prob)
        all_y.append(target.detach().cpu().numpy())
        if collect:
            all_id.extend(batch["id"])
            all_feat.append(out["feature"].detach().cpu().numpy())
        total += float(loss.item()) * mri.size(0)
    y = np.concatenate(all_y)
    prob = np.concatenate(all_prob)
    stats = metrics(y, prob)
    stats["loss"] = total / len(loader.dataset)
    outputs = None
    if collect:
        feat = np.concatenate(all_feat, axis=0)
        outputs = pd.DataFrame({"id": all_id, "y_true": y.astype(int)})
        for i in range(prob.shape[1]):
            outputs[f"prob_{i}"] = prob[:, i]
        if prob.shape[1] == 2:
            outputs["prob_positive"] = prob[:, 1]
        outputs = pd.concat([outputs, pd.DataFrame(feat, columns=[f"feat_{i:03d}" for i in range(feat.shape[1])])], axis=1)
    return stats, outputs


def run_student(teacher, student, loader, optimizer, device, args, weight=None, collect=False, debug_shapes=False):
    train = optimizer is not None
    teacher.eval()
    student.train(train)
    total = 0.0
    all_y, all_prob, all_id, all_feat = [], [], [], []
    debug_once = debug_shapes
    for batch in tqdm(loader, leave=False, desc="student" if train else "eval"):
        mri, pet, target = batch["mri"].to(device), batch["pet"].to(device), batch["target"].to(device)
        with torch.no_grad():
            tout = teacher(mri, pet)
        with torch.set_grad_enabled(train):
            sout = student(mri, debug_shapes=debug_once)
            debug_once = False
            ce = F.cross_entropy(sout["logits"], target, weight=weight)
            high = relation_kd_loss(tout["feature"], sout["feature"], args.relation_temperature)
            soft = soft_label_kd_loss(tout["logits"], sout["logits"], args.kd_temperature)
            attn = pf_cmad_loss(tout, sout)
            gate = smooth_confidence(tout["logits"], sout["logits"], target)
            loss = args.ce_weight * ce + args.kd_weight * gate * (high + soft + attn)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        prob = torch.softmax(sout["logits"], dim=1).detach().cpu().numpy()
        all_prob.append(prob)
        all_y.append(target.detach().cpu().numpy())
        if collect:
            all_id.extend(batch["id"])
            all_feat.append(sout["feature"].detach().cpu().numpy())
        total += float(loss.item()) * mri.size(0)
    y = np.concatenate(all_y)
    prob = np.concatenate(all_prob)
    stats = metrics(y, prob)
    stats["loss"] = total / len(loader.dataset)
    outputs = None
    if collect:
        feat = np.concatenate(all_feat, axis=0)
        outputs = pd.DataFrame({"id": all_id, "y_true": y.astype(int)})
        for i in range(prob.shape[1]):
            outputs[f"prob_{i}"] = prob[:, i]
        if prob.shape[1] == 2:
            outputs["prob_positive"] = prob[:, 1]
        outputs = pd.concat([outputs, pd.DataFrame(feat, columns=[f"feat_{i:03d}" for i in range(feat.shape[1])])], axis=1)
    return stats, outputs


def score(stats, metric):
    return stats.get(metric, stats.get("auc", stats.get("acc", -1.0)))


def train_fold(args, frame, train_idx, val_idx, fold, label_to_idx):
    train_ds = CRADDataset(frame.iloc[train_idx], args.input_shape, True)
    val_ds = CRADDataset(frame.iloc[val_idx], args.input_shape, False)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    teacher = CRADTeacher(len(label_to_idx), args.channels, args.feature_dim).to(device)
    student = CRADStudent(len(label_to_idx), args.channels, args.feature_dim).to(device)
    weight = class_weights(frame.iloc[train_idx]["target"].to_numpy(), len(label_to_idx), device)
    opt_t = torch.optim.AdamW(teacher.parameters(), lr=args.teacher_lr, weight_decay=args.weight_decay)
    opt_s = torch.optim.AdamW(student.parameters(), lr=args.student_lr, weight_decay=args.weight_decay)
    out_dir = Path(args.output_dir) / args.task / f"fold_{fold:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_teacher = {"auc": -1.0}
    best_teacher_state = None
    for epoch in range(1, args.teacher_epochs + 1):
        tr, _ = run_teacher(teacher, train_loader, opt_t, device, args, weight, debug_shapes=args.debug_shapes and epoch == 1)
        va, pred = run_teacher(teacher, val_loader, None, device, args, weight, collect=True)
        print(json.dumps({"phase": "teacher", "epoch": epoch, "train": tr, "val": va}, ensure_ascii=False))
        if score(va, args.select_metric) > score(best_teacher, args.select_metric):
            best_teacher = va
            best_teacher_state = copy.deepcopy(teacher.state_dict())
            torch.save({"teacher": best_teacher_state, "args": vars(args), "labels": label_to_idx, "metrics": va}, out_dir / "best_teacher.pt")
            pred.to_csv(out_dir / "teacher_val_predictions.csv", index=False)
    if best_teacher_state is not None:
        teacher.load_state_dict(best_teacher_state)

    best_student = {"auc": -1.0}
    for epoch in range(1, args.student_epochs + 1):
        tr, _ = run_student(teacher, student, train_loader, opt_s, device, args, weight, debug_shapes=args.debug_shapes and epoch == 1)
        va, pred = run_student(teacher, student, val_loader, None, device, args, weight, collect=True)
        print(json.dumps({"phase": "student", "epoch": epoch, "train": tr, "val": va}, ensure_ascii=False))
        if score(va, args.select_metric) > score(best_student, args.select_metric):
            best_student = va
            torch.save({"teacher": teacher.state_dict(), "student": student.state_dict(), "args": vars(args), "labels": label_to_idx, "metrics": va}, out_dir / "best.pt")
            pred.to_csv(out_dir / "val_predictions.csv", index=False)
    result = {"teacher": best_teacher, "student": best_student}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def summarize(results):
    if all(isinstance(v, dict) for v in results[0].values()):
        return {key: summarize([r[key] for r in results]) for key in results[0]}
    values = {k: [float(r[k]) for r in results if k in r] for k in results[0]}
    count_keys = {"tn", "fp", "fn", "tp"}
    return {k: ({"sum": int(np.sum(v))} if k in count_keys else {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))}) for k, v in values.items()}


def main():
    parser = argparse.ArgumentParser(description="Reproduce CRAD for MRI-only AD diagnosis with multimodal MRI/PET distillation.")
    parser.add_argument("--data-root", default=r"D:\AD\心睿\ADNI-amyloid-smri-pet")
    parser.add_argument("--mri-folder", default="mwp1")
    parser.add_argument("--pet-folder", default="pet")
    parser.add_argument("--task", default="AD_CN", choices=["AD_CN", "MCI_CN", "AD_MCI", "CN_MCI_AD"])
    parser.add_argument("--output-dir", default="crad_runs")
    parser.add_argument("--input-shape", type=parse_shape, default=(64, 80, 64))
    parser.add_argument("--channels", type=parse_channels, default=(16, 32, 64))
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--teacher-epochs", type=int, default=20)
    parser.add_argument("--student-epochs", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--single-split", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gamma-mmse", type=float, default=0.1)
    parser.add_argument("--gamma-orth", type=float, default=0.1)
    parser.add_argument("--ce-weight", type=float, default=0.5)
    parser.add_argument("--kd-weight", type=float, default=0.5)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--relation-temperature", type=float, default=1.0)
    parser.add_argument("--select-metric", default="auc")
    parser.add_argument("--teacher-lr", type=float, default=2e-4)
    parser.add_argument("--student-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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
        frame = pd.concat([g.sample(min(len(g), per_class), random_state=args.seed) for _, g in frame.groupby("target")]).reset_index(drop=True)
    print(f"Loaded {len(frame)} samples for {args.task}: {label_to_idx}")
    print(frame["Label"].value_counts().to_string())
    y = frame["target"].to_numpy()
    if args.single_split:
        val_fraction = max(0.2, frame["target"].nunique() / len(frame))
        train_idx, val_idx = train_test_split(np.arange(len(frame)), test_size=val_fraction, stratify=y, random_state=args.seed)
        results = [train_fold(args, frame, train_idx, val_idx, 0, label_to_idx)]
    else:
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        results = [train_fold(args, frame, train_idx, val_idx, fold, label_to_idx) for fold, (train_idx, val_idx) in enumerate(splitter.split(frame, y))]
    out_dir = Path(args.output_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

