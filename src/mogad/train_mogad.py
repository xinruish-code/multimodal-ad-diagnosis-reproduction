import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from mogad.data import MOGADDataset, collate, read_manifest
from mogad.model import MOGADStudent, MOGADTeacher, hfa_loss, hkd_loss, label_consistency_loss


def parse_shape(value):
    return tuple(int(v) for v in value.split(","))


def metrics(y_true, prob):
    pred = prob.argmax(axis=1)
    result = {"acc": accuracy_score(y_true, pred)}
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
    else:
        result["pre"] = precision_score(y_true, pred, average="macro", zero_division=0)
        result["f1"] = f1_score(y_true, pred, average="macro", zero_division=0)
        result["auc"] = roc_auc_score(y_true, prob, multi_class="ovr") if len(np.unique(y_true)) > 2 else float("nan")
    return result


def run_pretrain_epoch(teacher, loader, optimizer, device, debug_shapes=False):
    train = optimizer is not None
    teacher.train(train)
    total = 0.0
    debug_once = debug_shapes
    iterator = tqdm(loader, leave=False, desc="pretrain" if train else "pre_eval")
    for batch in iterator:
        brain = batch["brain_pet"].to(device)
        target = batch["target"].to(device)
        with torch.set_grad_enabled(train):
            features, final = teacher.brain(brain)
            logits = teacher.brain_head(final)
            if debug_once:
                print("pretrain_brain:", brain.shape, "features:", [f.shape for f in features], "logits:", logits.shape)
                debug_once = False
            loss = F.cross_entropy(logits, target)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total += float(loss.item()) * brain.size(0)
        iterator.set_postfix(loss=float(loss.item()))
    return {"loss": total / len(loader.dataset)}


def run_teacher_epoch(teacher, loader, optimizer, device, args, debug_shapes=False):
    train = optimizer is not None
    teacher.train(train)
    total = 0.0
    debug_once = debug_shapes
    iterator = tqdm(loader, leave=False, desc="teacher" if train else "teacher_eval")
    for batch in iterator:
        brain = batch["brain_pet"].to(device)
        aux1 = batch["aux1"].to(device)
        aux2 = batch["aux2"].to(device)
        target = batch["target"].to(device)
        with torch.set_grad_enabled(train):
            out = teacher(brain, aux1, aux2, debug_shapes=debug_once)
            debug_once = False
            with torch.no_grad():
                pseudo = out["brain_logits"].argmax(dim=1)
                conf = torch.softmax(out["brain_logits"], dim=1).max(dim=1).values
                keep = conf >= conf.median()
            pseudo_loss = F.cross_entropy(out["aux1_logits"][keep], pseudo[keep]) + F.cross_entropy(out["aux2_logits"][keep], pseudo[keep])
            ce = F.cross_entropy(out["logits"], target)
            loss = args.alpha * ce + args.gamma * pseudo_loss + args.hfa_weight * hfa_loss(out, args.temperature) + args.lc_weight * label_consistency_loss(out, target, args.temperature)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total += float(loss.item()) * brain.size(0)
        iterator.set_postfix(loss=float(loss.item()))
    return {"loss": total / len(loader.dataset)}


def run_student_epoch(teacher, student, loader, optimizer, device, args, collect_outputs=False, debug_shapes=False):
    train = optimizer is not None
    teacher.eval()
    student.train(train)
    all_y, all_prob, all_id, all_feat = [], [], [], []
    total = 0.0
    debug_once = debug_shapes
    iterator = tqdm(loader, leave=False, desc="student" if train else "eval")
    for batch in iterator:
        brain = batch["brain_pet"].to(device)
        aux1 = batch["aux1"].to(device)
        aux2 = batch["aux2"].to(device)
        target = batch["target"].to(device)
        with torch.no_grad():
            teacher_out = teacher(brain, aux1, aux2)
        with torch.set_grad_enabled(train):
            student_out = student(brain, debug_shapes=debug_once)
            debug_once = False
            ce = F.cross_entropy(student_out["logits"], target)
            kd = hkd_loss(student_out, teacher_out, student, args.temperature)
            loss = args.student_ce_weight * ce + args.kd_weight * kd
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        prob = torch.softmax(student_out["logits"], dim=1).detach().cpu().numpy()
        all_prob.append(prob)
        all_y.append(target.detach().cpu().numpy())
        if collect_outputs:
            all_feat.append(student_out["final"].detach().cpu().numpy())
            all_id.extend(batch["id"])
        total += float(loss.item()) * brain.size(0)
        iterator.set_postfix(loss=float(loss.item()))
    y = np.concatenate(all_y)
    prob = np.concatenate(all_prob)
    result = metrics(y, prob)
    result["loss"] = total / len(loader.dataset)
    outputs = None
    if collect_outputs:
        feat = np.concatenate(all_feat, axis=0)
        outputs = pd.DataFrame({"id": all_id, "y_true": y.astype(int)})
        for cls_idx in range(prob.shape[1]):
            outputs[f"prob_{cls_idx}"] = prob[:, cls_idx]
        if prob.shape[1] == 2:
            outputs["prob_positive"] = prob[:, 1]
        outputs = pd.concat([outputs, pd.DataFrame(feat, columns=[f"feat_{i:03d}" for i in range(feat.shape[1])])], axis=1)
    return result, outputs


def train_fold(args, frame, train_idx, val_idx, fold, label_to_idx):
    train_ds = MOGADDataset(frame.iloc[train_idx], args.input_shape, augment=True)
    val_ds = MOGADDataset(frame.iloc[val_idx], args.input_shape, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    teacher = MOGADTeacher(len(label_to_idx), args.patch_size, args.embed_dim, args.dropout).to(device)
    student = MOGADStudent(len(label_to_idx), args.patch_size, args.embed_dim, args.dropout).to(device)
    opt_pre = torch.optim.AdamW(list(teacher.brain.parameters()) + list(teacher.brain_head.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    opt_teacher = torch.optim.AdamW(teacher.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_student = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_dir = Path(args.output_dir) / args.task / f"fold_{fold:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.pretrain_epochs + 1):
        stats = run_pretrain_epoch(teacher, train_loader, opt_pre, device, args.debug_shapes and epoch == 1)
        print(json.dumps({"phase": "pretrain", "epoch": epoch, "train": stats}, ensure_ascii=False))

    for epoch in range(1, args.teacher_epochs + 1):
        stats = run_teacher_epoch(teacher, train_loader, opt_teacher, device, args, args.debug_shapes and epoch == 1)
        print(json.dumps({"phase": "teacher", "epoch": epoch, "train": stats}, ensure_ascii=False))

    best = {"auc": -1.0}
    for epoch in range(1, args.student_epochs + 1):
        train_stats, _ = run_student_epoch(teacher, student, train_loader, opt_student, device, args, debug_shapes=args.debug_shapes and epoch == 1)
        val_stats, val_outputs = run_student_epoch(teacher, student, val_loader, None, device, args, collect_outputs=True)
        print(json.dumps({"phase": "student", "epoch": epoch, "train": train_stats, "val": val_stats}, ensure_ascii=False))
        score = val_stats.get("auc", val_stats["acc"])
        if score > best.get("auc", -1.0):
            best = val_stats
            torch.save({"teacher": teacher.state_dict(), "student": student.state_dict(), "args": vars(args), "labels": label_to_idx, "metrics": best}, out_dir / "best.pt")
            val_outputs.to_csv(out_dir / "val_predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    return best


def main():
    parser = argparse.ArgumentParser(description="Reproduce MOGAD-Net with local brain-only ADNI-style data adaptation.")
    parser.add_argument("--data-root", default=r"D:\AD\心睿\ADNI-amyloid-smri-pet")
    parser.add_argument("--brain-pet-folder", default="pet")
    parser.add_argument("--aux1-folder", default="mwp1", help="Original heart PET branch adapted to GM/MRI in local data.")
    parser.add_argument("--aux2-folder", default="wm", help="Original gut PET branch adapted to WM in local data.")
    parser.add_argument("--task", default="MCI_CN", choices=["AD_CN", "MCI_CN", "AD_MCI", "CN_MCI_AD"])
    parser.add_argument("--output-dir", default="mogad_runs")
    parser.add_argument("--input-shape", type=parse_shape, default=(64, 80, 64))
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--teacher-epochs", type=int, default=10)
    parser.add_argument("--student-epochs", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--single-split", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--hfa-weight", type=float, default=0.1)
    parser.add_argument("--lc-weight", type=float, default=0.1)
    parser.add_argument("--student-ce-weight", type=float, default=1.0)
    parser.add_argument("--kd-weight", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-shapes", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    frame, label_to_idx = read_manifest(args.data_root, args.task, args.brain_pet_folder, args.aux1_folder, args.aux2_folder)
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
    summary = {k: [float(r[k]) for r in results] for k in results[0]}
    count_keys = {"tn", "fp", "fn", "tp"}
    summary = {
        k: ({"sum": int(np.sum(v))} if k in count_keys else {"mean": float(np.mean(v)), "std": float(np.std(v))})
        for k, v in summary.items()
    }
    out_dir = Path(args.output_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
