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

from mdl_net.data import ADNIMultimodalDataset, load_roi_table, read_manifest
from mdl_net.model import MDLNet


def parse_shape(value):
    return tuple(int(v) for v in value.split(","))


def metrics(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "acc": accuracy_score(y_true, pred),
        "sen": tp / max(tp + fn, 1),
        "spec": tn / max(tn + fp, 1),
        "pre": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc": roc_auc_score(y_true, prob) if len(np.unique(y_true)) == 2 else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def make_scheduler(optimizer, warmup_epochs, epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def collate(batch):
    out = {
        "image": torch.stack([b["image"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "id": [b["id"] for b in batch],
    }
    if all("roi" in b for b in batch):
        out["roi"] = torch.stack([b["roi"] for b in batch])
    return out


def run_epoch(model, loader, optimizer, device, label_smoothing, roi_weight, debug_shapes=False, collect_outputs=False):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    all_y, all_prob = [], []
    all_id, all_feat = [], []
    debug_once = debug_shapes
    iterator = tqdm(loader, leave=False, desc="train" if train else "eval")
    for batch in iterator:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            out = model(image, debug_shapes=debug_once)
            debug_once = False
            loss = F.cross_entropy(out["logits"], target, label_smoothing=label_smoothing if train else 0.0)
            if "roi" in batch and "roi_pred" in out:
                roi = batch["roi"].to(device, non_blocking=True)
                loss = loss + roi_weight * F.mse_loss(out["roi_pred"], roi)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        prob = torch.softmax(out["logits"], dim=1)[:, 1].detach().cpu().numpy()
        all_prob.append(prob)
        all_y.append(target.detach().cpu().numpy())
        if collect_outputs:
            all_id.extend(batch["id"])
            all_feat.append(out["features"].detach().cpu().numpy())
        total_loss += float(loss.item()) * image.size(0)
        iterator.set_postfix(loss=float(loss.item()))
    y = np.concatenate(all_y)
    prob = np.concatenate(all_prob)
    result = metrics(y, prob)
    result["loss"] = total_loss / len(loader.dataset)
    outputs = None
    if collect_outputs:
        features = np.concatenate(all_feat, axis=0)
        meta_outputs = pd.DataFrame({"id": all_id, "y_true": y.astype(int), "prob_positive": prob})
        feature_outputs = pd.DataFrame(features, columns=[f"feat_{i:03d}" for i in range(features.shape[1])])
        outputs = pd.concat([meta_outputs, feature_outputs], axis=1)
    return result, outputs


def train_fold(args, frame, train_idx, val_idx, fold, label_info, roi_table):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_ds = ADNIMultimodalDataset(frame.iloc[train_idx], args.input_shape, augment=True, roi_table=roi_table)
    val_ds = ADNIMultimodalDataset(frame.iloc[val_idx], args.input_shape, augment=False, roi_table=roi_table)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate, pin_memory=True)

    use_drl = roi_table is not None and args.use_drl
    model = MDLNet(use_drl=use_drl, drl_iterations=args.drl_iterations).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.warmup_epochs, args.epochs)

    best = {"auc": -1.0}
    out_dir = Path(args.output_dir) / args.task / f"fold_{fold:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_stats, _ = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.label_smoothing,
            args.roi_weight,
            debug_shapes=args.debug_shapes and epoch == 0,
        )
        val_stats, val_outputs = run_epoch(model, val_loader, None, device, 0.0, args.roi_weight, collect_outputs=True)
        scheduler.step()
        row = {"epoch": epoch + 1, "lr": scheduler.get_last_lr()[0], "train": train_stats, "val": val_stats}
        print(json.dumps(row, ensure_ascii=False))
        if val_stats["auc"] > best["auc"]:
            best = val_stats
            torch.save({"model": model.state_dict(), "args": vars(args), "label_info": label_info, "metrics": best}, out_dir / "best.pt")
            val_outputs.to_csv(out_dir / "val_predictions.csv", index=False)

    (out_dir / "metrics.json").write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    return best


def main():
    parser = argparse.ArgumentParser(description="Reproduce MDL-Net on local ADNI amyloid sMRI PET data.")
    parser.add_argument("--data-root", default=r"D:\AD\心睿\ADNI-amyloid-smri-pet")
    parser.add_argument("--task", default="AD_CN", choices=["AD_CN", "MCI_CN", "AD_MCI"])
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--input-shape", type=parse_shape, default=(96, 112, 96), help="D,H,W, e.g. 96,112,96")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--single-split", action="store_true", help="Use one stratified 80/20 split for a quick smoke run.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional stratified subset size for debugging.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--roi-csv", default=None, help="Optional CSV keyed by ID with 90 ROI columns.")
    parser.add_argument("--use-drl", action="store_true", help="Enable disease-induced ROI learning when --roi-csv is provided.")
    parser.add_argument("--drl-iterations", type=int, default=3)
    parser.add_argument("--roi-weight", type=float, default=0.5)
    parser.add_argument("--debug-shapes", action="store_true", help="Print MDL-Net tensor shapes for the first training batch.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    frame, label_info = read_manifest(args.data_root, args.task)
    if args.max_samples is not None and args.max_samples < len(frame):
        per_class = max(1, args.max_samples // frame["target"].nunique())
        pieces = []
        for _, group in frame.groupby("target"):
            pieces.append(group.sample(min(len(group), per_class), random_state=args.seed))
        frame = frame.iloc[0:0].copy() if not pieces else pd.concat(pieces).reset_index(drop=True)
    roi_table = load_roi_table(args.roi_csv)
    print(f"Loaded {len(frame)} samples for {args.task}: {label_info}")
    print(frame["Label"].value_counts().to_string())

    y = frame["target"].to_numpy()
    if args.single_split:
        train_idx, val_idx = train_test_split(np.arange(len(frame)), test_size=0.2, stratify=y, random_state=args.seed)
        results = [train_fold(args, frame, train_idx, val_idx, 0, label_info, roi_table)]
    else:
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        results = [train_fold(args, frame, train_idx, val_idx, fold, label_info, roi_table) for fold, (train_idx, val_idx) in enumerate(splitter.split(frame, y))]

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
