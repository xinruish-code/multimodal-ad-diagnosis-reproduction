import argparse
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

from aagn.data import AAGNDataset, collate, read_manifest
from aagn.model import AAGN
from aagn.roi import build_roi_masks


def parse_shape(value):
    return tuple(int(v) for v in value.split(","))


def metrics(y_true, prob):
    pred = prob.argmax(axis=1)
    result = {"acc": accuracy_score(y_true, pred), "bal_acc": balanced_accuracy_score(y_true, pred)}
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
        result["f1"] = f1_score(y_true, pred, average="macro", zero_division=0)
        result["auc"] = roc_auc_score(y_true, prob, multi_class="ovr") if len(np.unique(y_true)) > 2 else float("nan")
    return result


def run_epoch(model, loader, optimizer, device, collect_outputs=False, debug_shapes=False):
    train = optimizer is not None
    model.train(train)
    all_y, all_prob, all_id, all_feat, all_gate = [], [], [], [], []
    total_loss = 0.0
    debug_once = debug_shapes
    iterator = tqdm(loader, leave=False, desc="train" if train else "eval")
    for batch in iterator:
        image = batch["image"].to(device)
        target = batch["target"].to(device)
        with torch.set_grad_enabled(train):
            out = model(image, debug_shapes=debug_once)
            debug_once = False
            loss = F.cross_entropy(out["logits"], target)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        prob = torch.softmax(out["logits"], dim=1).detach().cpu().numpy()
        all_prob.append(prob)
        all_y.append(target.detach().cpu().numpy())
        if collect_outputs:
            all_feat.append(out["features"].detach().cpu().numpy())
            all_gate.append(out["gate_prob"].detach().cpu().numpy())
            all_id.extend(batch["id"])
        total_loss += float(loss.item()) * image.size(0)
        iterator.set_postfix(loss=float(loss.item()))
    y = np.concatenate(all_y)
    prob = np.concatenate(all_prob)
    result = metrics(y, prob)
    result["loss"] = total_loss / len(loader.dataset)
    outputs = None
    gates = None
    if collect_outputs:
        feat = np.concatenate(all_feat, axis=0)
        gate = np.concatenate(all_gate, axis=0)
        outputs = pd.DataFrame({"id": all_id, "y_true": y.astype(int)})
        for cls_idx in range(prob.shape[1]):
            outputs[f"prob_{cls_idx}"] = prob[:, cls_idx]
        if prob.shape[1] == 2:
            outputs["prob_positive"] = prob[:, 1]
        feat_df = pd.DataFrame(feat, columns=[f"feat_{i:03d}" for i in range(feat.shape[1])])
        outputs = pd.concat([outputs, feat_df], axis=1)
        gates = pd.DataFrame(gate, columns=[f"roi_{i:03d}" for i in range(gate.shape[1])])
        gates.insert(0, "id", all_id)
        gates.insert(1, "y_true", y.astype(int))
    return result, outputs, gates


def train_fold(args, frame, train_idx, val_idx, fold, label_to_idx, roi_masks):
    train_ds = AAGNDataset(frame.iloc[train_idx], args.input_shape, augment=True)
    val_ds = AAGNDataset(frame.iloc[val_idx], args.input_shape, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = AAGN(
        roi_masks=roi_masks,
        num_classes=len(label_to_idx),
        channels=args.channels,
        blocks=args.blocks,
        embed_dim=args.embed_dim,
        tau=args.tau,
        hard=args.hard_gate,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_dir = Path(args.output_dir) / args.task / f"fold_{fold:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best = {"auc": -1.0}
    for epoch in range(1, args.epochs + 1):
        train_stats, _, _ = run_epoch(model, train_loader, optimizer, device, debug_shapes=args.debug_shapes and epoch == 1)
        val_stats, val_outputs, val_gates = run_epoch(model, val_loader, None, device, collect_outputs=True)
        row = {"epoch": epoch, "train": train_stats, "val": val_stats}
        print(json.dumps(row, ensure_ascii=False))
        score = val_stats.get("auc", val_stats["bal_acc"])
        if score > best.get("auc", -1.0):
            best = val_stats
            torch.save({"model": model.state_dict(), "args": vars(args), "labels": label_to_idx, "metrics": best}, out_dir / "best.pt")
            val_outputs.to_csv(out_dir / "val_predictions.csv", index=False)
            val_gates.to_csv(out_dir / "val_roi_gates.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    return best


def main():
    parser = argparse.ArgumentParser(description="Reproduce AAGN on local ADNI sMRI data.")
    parser.add_argument("--data-root", default=r"D:\AD\心睿\ADNI-amyloid-smri-pet")
    parser.add_argument("--image-folder", default="mwp1")
    parser.add_argument("--atlas-path", default=None, help="Optional ROI label atlas NIfTI. If omitted, grid ROIs are used.")
    parser.add_argument("--roi-grid", type=parse_shape, default=(4, 4, 4))
    parser.add_argument("--task", default="AD_CN", choices=["AD_CN", "MCI_CN", "AD_MCI", "CN_MCI_AD"])
    parser.add_argument("--output-dir", default="aagn_runs")
    parser.add_argument("--input-shape", type=parse_shape, default=(64, 80, 64))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--single-split", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--hard-gate", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-shapes", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    frame, label_to_idx = read_manifest(args.data_root, args.task, args.image_folder)
    if args.max_samples is not None and args.max_samples < len(frame):
        per_class = max(1, args.max_samples // frame["target"].nunique())
        pieces = [group.sample(min(len(group), per_class), random_state=args.seed) for _, group in frame.groupby("target")]
        frame = pd.concat(pieces).reset_index(drop=True)
    roi_masks = build_roi_masks(args.input_shape, args.atlas_path, args.roi_grid)

    print(f"Loaded {len(frame)} samples for {args.task}: {label_to_idx}")
    print(f"ROI masks: {tuple(roi_masks.shape)}")
    print(frame["Label"].value_counts().to_string())
    y = frame["target"].to_numpy()
    if args.single_split:
        train_idx, val_idx = train_test_split(np.arange(len(frame)), test_size=0.2, stratify=y, random_state=args.seed)
        results = [train_fold(args, frame, train_idx, val_idx, 0, label_to_idx, roi_masks)]
    else:
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        results = [train_fold(args, frame, train_idx, val_idx, fold, label_to_idx, roi_masks) for fold, (train_idx, val_idx) in enumerate(splitter.split(frame, y))]

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
