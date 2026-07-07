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

from unicross.data import UniCrossDataset, collate, read_manifest
from unicross.model import UniCross, metadata_weighted_contrastive


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
        result["pre"] = precision_score(y_true, pred, average="macro", zero_division=0)
        result["f1"] = f1_score(y_true, pred, average="macro", zero_division=0)
        result["auc"] = roc_auc_score(y_true, prob, multi_class="ovr") if len(np.unique(y_true)) > 2 else float("nan")
    return result


def run_encoder_epoch(model, loader, optimizer, device, args, debug_shapes=False):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    debug_once = debug_shapes
    iterator = tqdm(loader, leave=False, desc="enc_train" if train else "enc_eval")
    for batch in iterator:
        mri = batch["mri"].to(device)
        pet = batch["pet"].to(device)
        meta = batch["meta"].to(device)
        target = batch["target"].to(device)
        with torch.set_grad_enabled(train):
            out = model.forward_encoder_stage(mri, pet, meta, debug_shapes=debug_once)
            debug_once = False
            l_uni = F.cross_entropy(out["mri_logits"], target) + F.cross_entropy(out["pet_logits"], target)
            l_sp = F.cross_entropy(out["shared_mri_logits"], target) + F.cross_entropy(out["shared_pet_logits"], target)
            l_mwcl = metadata_weighted_contrastive(out["f_mri"], out["f_pet"], out["f_meta"], target, args.temperature)
            loss = l_uni + l_sp + args.mwcl_weight * l_mwcl
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * mri.size(0)
        iterator.set_postfix(loss=float(loss.item()))
    return {"loss": total_loss / len(loader.dataset)}


def run_fusion_epoch(model, loader, optimizer, device, collect_outputs=False, debug_shapes=False):
    train = optimizer is not None
    model.train(train)
    all_y, all_prob, all_id, all_feat = [], [], [], []
    total_loss = 0.0
    debug_once = debug_shapes
    iterator = tqdm(loader, leave=False, desc="fusion_train" if train else "fusion_eval")
    for batch in iterator:
        mri = batch["mri"].to(device)
        pet = batch["pet"].to(device)
        meta = batch["meta"].to(device)
        target = batch["target"].to(device)
        with torch.set_grad_enabled(train):
            out = model.forward_fusion_stage(mri, pet, meta, debug_shapes=debug_once)
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
            all_id.extend(batch["id"])
        total_loss += float(loss.item()) * mri.size(0)
        iterator.set_postfix(loss=float(loss.item()))
    y = np.concatenate(all_y)
    prob = np.concatenate(all_prob)
    result = metrics(y, prob)
    result["loss"] = total_loss / len(loader.dataset)
    outputs = None
    if collect_outputs:
        feat = np.concatenate(all_feat, axis=0)
        outputs = pd.DataFrame({"id": all_id, "y_true": y.astype(int)})
        for cls_idx in range(prob.shape[1]):
            outputs[f"prob_{cls_idx}"] = prob[:, cls_idx]
        if prob.shape[1] == 2:
            outputs["prob_positive"] = prob[:, 1]
        feat_df = pd.DataFrame(feat, columns=[f"feat_{i:03d}" for i in range(feat.shape[1])])
        outputs = pd.concat([outputs, feat_df], axis=1)
    return result, outputs


def train_fold(args, frame, train_idx, val_idx, fold, label_to_idx, meta_dim):
    train_ds = UniCrossDataset(frame.iloc[train_idx], args.input_shape, augment=True)
    val_ds = UniCrossDataset(frame.iloc[val_idx], args.input_shape, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    max_tokens = int(np.prod([s // args.patch_size for s in args.input_shape]))
    model = UniCross(meta_dim, len(label_to_idx), args.patch_size, args.embed_dim, args.depth, args.heads, args.dropout, max_tokens).to(device)
    enc_params = list(model.mri_encoder.parameters()) + list(model.pet_encoder.parameters()) + list(model.meta_encoder.parameters()) + list(model.mri_head.parameters()) + list(model.pet_head.parameters()) + list(model.shared_head.parameters())
    enc_optimizer = torch.optim.AdamW(enc_params, lr=args.encoder_lr, weight_decay=args.weight_decay)
    enc_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(enc_optimizer, T_0=10, T_mult=3, eta_min=1e-5)

    out_dir = Path(args.output_dir) / args.task / f"fold_{fold:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.encoder_epochs + 1):
        train_stats = run_encoder_epoch(model, train_loader, enc_optimizer, device, args, args.debug_shapes and epoch == 1)
        enc_scheduler.step()
        print(json.dumps({"phase": "encoder", "epoch": epoch, "lr": enc_scheduler.get_last_lr()[0], "train": train_stats}, ensure_ascii=False))

    for module in (model.mri_encoder, model.pet_encoder, model.meta_encoder, model.mri_head, model.pet_head, model.shared_head):
        for param in module.parameters():
            param.requires_grad = False
    fusion_optimizer = torch.optim.Adam(model.fusion_head.parameters(), lr=args.fusion_lr, weight_decay=args.weight_decay)
    best = {"auc": -1.0}
    for epoch in range(1, args.fusion_epochs + 1):
        train_stats, _ = run_fusion_epoch(model, train_loader, fusion_optimizer, device, debug_shapes=args.debug_shapes and epoch == 1)
        val_stats, val_outputs = run_fusion_epoch(model, val_loader, None, device, collect_outputs=True)
        row = {"phase": "fusion", "epoch": epoch, "train": train_stats, "val": val_stats}
        print(json.dumps(row, ensure_ascii=False))
        score = val_stats.get("auc", val_stats["bal_acc"])
        if score > best.get("auc", -1.0):
            best = val_stats
            torch.save({"model": model.state_dict(), "args": vars(args), "labels": label_to_idx, "metrics": best}, out_dir / "best.pt")
            val_outputs.to_csv(out_dir / "val_predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    return best


def main():
    parser = argparse.ArgumentParser(description="Reproduce UniCross on local ADNI MRI/PET data.")
    parser.add_argument("--data-root", default=r"D:\AD\心睿\ADNI-amyloid-smri-pet")
    parser.add_argument("--mri-folder", default="mwp1")
    parser.add_argument("--pet-folder", default="pet")
    parser.add_argument("--meta-columns", default="Age,Sex,MMSE Total Score")
    parser.add_argument("--task", default="AD_CN", choices=["AD_CN", "MCI_CN", "AD_MCI", "CN_MCI_AD"])
    parser.add_argument("--output-dir", default="unicross_runs")
    parser.add_argument("--input-shape", type=parse_shape, default=(64, 80, 64))
    parser.add_argument("--encoder-epochs", type=int, default=40)
    parser.add_argument("--fusion-epochs", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--single-split", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--mwcl-weight", type=float, default=1.0)
    parser.add_argument("--encoder-lr", type=float, default=5e-4)
    parser.add_argument("--fusion-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-shapes", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    meta_columns = [c.strip() for c in args.meta_columns.split(",") if c.strip()]
    frame, label_to_idx, used_meta = read_manifest(args.data_root, args.task, args.mri_folder, args.pet_folder, meta_columns)
    if args.max_samples is not None and args.max_samples < len(frame):
        per_class = max(1, args.max_samples // frame["target"].nunique())
        pieces = [group.sample(min(len(group), per_class), random_state=args.seed) for _, group in frame.groupby("target")]
        frame = pd.concat(pieces).reset_index(drop=True)

    meta_dim = len([c for c in frame.columns if c.startswith("meta__")])
    print(f"Loaded {len(frame)} samples for {args.task}: {label_to_idx}")
    print(f"Metadata columns used: {used_meta} -> dim={meta_dim}")
    print(frame["Label"].value_counts().to_string())
    y = frame["target"].to_numpy()
    if args.single_split:
        train_idx, val_idx = train_test_split(np.arange(len(frame)), test_size=0.2, stratify=y, random_state=args.seed)
        results = [train_fold(args, frame, train_idx, val_idx, 0, label_to_idx, meta_dim)]
    else:
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        results = [train_fold(args, frame, train_idx, val_idx, fold, label_to_idx, meta_dim) for fold, (train_idx, val_idx) in enumerate(splitter.split(frame, y))]

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
