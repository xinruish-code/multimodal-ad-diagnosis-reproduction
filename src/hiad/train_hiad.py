import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from hiad.data import HIADDataset, collate, read_manifest
from hiad.model import HIAD, hiad_losses


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
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })
    else:
        result["sen"] = result["bca"]
        result["spec"] = result["bca"]
        result["pre"] = precision_score(y_true, pred, average="weighted", zero_division=0)
        result["f1"] = f1_score(y_true, pred, average="weighted", zero_division=0)
        result["auc"] = roc_auc_score(y_true, prob, multi_class="ovr") if len(np.unique(y_true)) > 2 else float("nan")
    return result


def run_epoch(model, loader, optimizer, device, args, epoch=1, collect_outputs=False, debug_shapes=False):
    train = optimizer is not None
    model.train(train)
    total = 0.0
    all_y, all_prob, all_id, all_feat, all_rho = [], [], [], [], []
    debug_once = debug_shapes
    iterator = tqdm(loader, leave=False, desc="train" if train else "eval")
    for batch in iterator:
        mri = batch["mri"].to(device)
        pet = batch["pet"].to(device)
        cad = batch["cad"].to(device)
        rho = batch["rho"].to(device)
        target = batch["target"].to(device)
        with torch.set_grad_enabled(train):
            out = model(mri, pet, cad, rho, epoch=epoch, train_mode=train, debug_shapes=debug_once)
            debug_once = False
            loss = hiad_losses(out, target, rho, args.lambda_intra, args.gamma_inter)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        prob = out["prob"].detach().cpu().numpy()
        all_prob.append(prob)
        all_y.append(target.detach().cpu().numpy())
        if collect_outputs:
            all_feat.append(out["z"].detach().cpu().numpy())
            all_rho.append(rho.detach().cpu().numpy())
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
        rho_np = np.concatenate(all_rho, axis=0)
        outputs = pd.DataFrame({"id": all_id, "y_true": y.astype(int)})
        for cls_idx in range(prob.shape[1]):
            outputs[f"prob_{cls_idx}"] = prob[:, cls_idx]
        if prob.shape[1] == 2:
            outputs["prob_positive"] = prob[:, 1]
        outputs = pd.concat([
            outputs,
            pd.DataFrame(feat, columns=[f"feat_{i:03d}" for i in range(feat.shape[1])]),
            pd.DataFrame(rho_np, columns=["rho_mri", "rho_pet", "rho_cad"]),
        ], axis=1)
    return result, outputs


def train_fold(args, frame, train_idx, val_idx, fold, label_to_idx, cad_dim):
    train_ds = HIADDataset(frame.iloc[train_idx], args.input_shape, args.missing_rate, False, args.seed + fold, augment=True)
    val_ds = HIADDataset(frame.iloc[val_idx], args.input_shape, args.missing_rate, True, args.seed + fold, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = HIAD(len(label_to_idx), cad_dim, args.base_channels, args.feature_dim, args.views, args.beta, args.tau).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_dir = Path(args.output_dir) / args.task / f"fold_{fold:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best = {"auc": -1.0}
    for epoch in range(1, args.epochs + 1):
        train_stats, _ = run_epoch(model, train_loader, optimizer, device, args, epoch, debug_shapes=args.debug_shapes and epoch == 1)
        val_stats, val_outputs = run_epoch(model, val_loader, None, device, args, epoch, collect_outputs=True)
        print(json.dumps({"epoch": epoch, "train": train_stats, "val": val_stats}, ensure_ascii=False))
        score = val_stats.get(args.select_metric, val_stats.get("auc", val_stats["acc"]))
        if score > best.get(args.select_metric, best.get("auc", -1.0)):
            best = val_stats
            torch.save({"model": model.state_dict(), "args": vars(args), "labels": label_to_idx, "metrics": best}, out_dir / "best.pt")
            val_outputs.to_csv(out_dir / "val_predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    return best


def summarize(results):
    summary = {k: [float(r[k]) for r in results if k in r] for k in results[0]}
    count_keys = {"tn", "fp", "fn", "tp"}
    return {
        k: ({"sum": int(np.sum(v))} if k in count_keys else {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))})
        for k, v in summary.items()
    }


def main():
    parser = argparse.ArgumentParser(description="Reproduce HI-AD with multi-view Hilbert-like hierarchical aggregation for incomplete multimodal AD diagnosis.")
    parser.add_argument("--data-root", default=r"D:\AD\心睿\ADNI-amyloid-smri-pet")
    parser.add_argument("--mri-folder", default="mwp1")
    parser.add_argument("--pet-folder", default="pet")
    parser.add_argument("--task", default="AD_CN", choices=["AD_CN", "MCI_CN", "AD_MCI", "CN_MCI_AD"])
    parser.add_argument("--output-dir", default="hiad_runs")
    parser.add_argument("--input-shape", type=parse_shape, default=(64, 80, 64))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--single-split", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--missing-rate", type=float, default=0.3)
    parser.add_argument("--lambda-intra", type=float, default=0.1)
    parser.add_argument("--gamma-inter", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.97)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--select-metric", default="auc")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-shapes", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    frame, label_to_idx, used_cad = read_manifest(args.data_root, args.task, args.mri_folder, args.pet_folder)
    if args.max_samples is not None and args.max_samples < len(frame):
        per_class = max(1, args.max_samples // frame["target"].nunique())
        pieces = [group.sample(min(len(group), per_class), random_state=args.seed) for _, group in frame.groupby("target")]
        frame = pd.concat(pieces).reset_index(drop=True)
    print(f"Loaded {len(frame)} samples for {args.task}: {label_to_idx}")
    print(f"CAD columns: {used_cad}")
    print(frame["Label"].value_counts().to_string())
    y = frame["target"].to_numpy()
    cad_dim = len([c for c in frame.columns if c.startswith("cad__")])
    if args.single_split:
        val_fraction = max(0.2, frame["target"].nunique() / len(frame))
        train_idx, val_idx = train_test_split(np.arange(len(frame)), test_size=val_fraction, stratify=y, random_state=args.seed)
        results = [train_fold(args, frame, train_idx, val_idx, 0, label_to_idx, cad_dim)]
    else:
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        results = [train_fold(args, frame, train_idx, val_idx, fold, label_to_idx, cad_dim) for fold, (train_idx, val_idx) in enumerate(splitter.split(frame, y))]
    out_dir = Path(args.output_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

