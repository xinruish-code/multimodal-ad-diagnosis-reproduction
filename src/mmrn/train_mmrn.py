import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from mmrn.data import MMRNDataset, collate, read_manifest
from mmrn.model import MMRN


def parse_shape(value):
    return tuple(int(v) for v in value.split(","))


def classification_metrics(y_true, prob):
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
        result["auc"] = roc_auc_score(y_true, prob, multi_class="ovr") if len(np.unique(y_true)) > 2 else float("nan")
    return result


def mmrn_losses(model, out, target, meta, weights):
    oi, oj = out["i"], out["j"]
    ce = F.cross_entropy(oi["logits"], target) + F.cross_entropy(oj["logits"], target)
    cs = -(F.cosine_similarity(oi["fc"], oj["fc"], dim=1).mean() + F.cosine_similarity(oi["fm"], oj["fm"], dim=1).mean())
    fake_fm = out["generated_fm"]
    real_adv = F.binary_cross_entropy_with_logits(model.discriminator(oi["fm"].detach()), torch.ones_like(model.discriminator(oi["fm"].detach())))
    fake_adv = F.binary_cross_entropy_with_logits(model.discriminator(fake_fm.detach()), torch.zeros_like(model.discriminator(fake_fm.detach())))
    disc = real_adv + fake_adv
    gen_adv = F.binary_cross_entropy_with_logits(model.discriminator(fake_fm), torch.ones_like(model.discriminator(fake_fm)))
    meta_pred = model.q_net(fake_fm)
    q_loss = F.mse_loss(meta_pred, meta)
    club = model.club(oi["fc"], oi["fm"])
    rec_i_fake = F.mse_loss(model.reconstructor(oi["fc"], fake_fm), oi["latent"].detach())
    rec_i_real = F.mse_loss(model.reconstructor(oi["fc"], oi["fm"]), oi["latent"].detach())
    rec_j_real = F.mse_loss(model.reconstructor(oj["fc"], oj["fm"]), oj["latent"].detach())
    rec = rec_i_fake + rec_i_real + rec_j_real
    total = weights["ce"] * ce + weights["ssl"] * cs + weights["gen"] * gen_adv + weights["q"] * q_loss + weights["club"] * club + weights["rec"] * rec + weights["disc"] * disc
    return total, {"ce": ce, "cs": cs, "disc": disc, "gen": gen_adv, "q": q_loss, "club": club, "rec": rec}


def run_epoch(model, loader, optimizer, device, weights, debug_shapes=False, collect_outputs=False):
    train = optimizer is not None
    model.train(train)
    all_y, all_prob = [], []
    all_id, all_fc = [], []
    total_loss = 0.0
    debug_once = debug_shapes
    iterator = tqdm(loader, leave=False, desc="train" if train else "eval")
    for batch in iterator:
        vi = batch["view_i"].to(device)
        vj = batch["view_j"].to(device)
        target = batch["target"].to(device)
        meta = batch["meta"].to(device)
        with torch.set_grad_enabled(train):
            out = model(vi, vj, meta, debug_shapes=debug_once)
            debug_once = False
            loss, parts = mmrn_losses(model, out, target, meta, weights)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        prob = torch.softmax(out["i"]["logits"], dim=1).detach().cpu().numpy()
        all_prob.append(prob)
        all_y.append(target.detach().cpu().numpy())
        if collect_outputs:
            all_id.extend(batch["id"])
            all_fc.append(out["i"]["fc"].detach().cpu().numpy())
        total_loss += float(loss.item()) * vi.size(0)
        iterator.set_postfix(loss=float(loss.item()))
    y = np.concatenate(all_y)
    prob = np.concatenate(all_prob)
    result = classification_metrics(y, prob)
    result["loss"] = total_loss / len(loader.dataset)
    outputs = None
    if collect_outputs:
        fc = np.concatenate(all_fc, axis=0)
        outputs = pd.DataFrame({"id": all_id, "y_true": y.astype(int)})
        for cls_idx in range(prob.shape[1]):
            outputs[f"prob_{cls_idx}"] = prob[:, cls_idx]
        if prob.shape[1] == 2:
            outputs["prob_positive"] = prob[:, 1]
        fc_outputs = pd.DataFrame(fc, columns=[f"fc_{i:03d}" for i in range(fc.shape[1])])
        outputs = pd.concat([outputs, fc_outputs], axis=1)
    return result, outputs


def main():
    parser = argparse.ArgumentParser(description="Reproduce MMRN-style sMRI/meta-information model on local ADNI data.")
    parser.add_argument("--data-root", default=r"D:\AD\心睿\ADNI-amyloid-smri-pet")
    parser.add_argument("--image-folder", default="mwp1", help="Use mwp1 GM by default; wm is also available.")
    parser.add_argument("--task", default="AD_CN", choices=["AD_CN", "MCI_CN", "AD_MCI", "CN_MCI_AD"])
    parser.add_argument("--meta-columns", default="Age,Sex,MMSE Total Score")
    parser.add_argument("--input-shape", type=parse_shape, default=(64, 80, 64))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="mmrn_runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-shapes", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    meta_cols = [c.strip() for c in args.meta_columns.split(",") if c.strip()]
    frame, metadata, label_to_idx = read_manifest(args.data_root, args.task, args.image_folder, meta_cols)
    if args.max_samples is not None and args.max_samples < len(frame):
        per_class = max(1, args.max_samples // frame["target"].nunique())
        keep = []
        for _, group in frame.groupby("target"):
            keep.append(group.sample(min(per_class, len(group)), random_state=args.seed).index)
        keep = np.concatenate([x.to_numpy() for x in keep])
        frame = frame.loc[keep].reset_index(drop=True)
        metadata = metadata.loc[keep].reset_index(drop=True)

    print(f"Loaded {len(frame)} samples for {args.task}: {label_to_idx}")
    print(frame["Label"].value_counts().to_string())
    print(f"Metadata columns: {metadata.columns.tolist()}")

    idx = np.arange(len(frame))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, stratify=frame["target"], random_state=args.seed)
    train_ds = MMRNDataset(frame.iloc[train_idx], metadata.iloc[train_idx], args.input_shape, augment=True)
    val_ds = MMRNDataset(frame.iloc[val_idx], metadata.iloc[val_idx], args.input_shape, augment=False)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = MMRN(meta_dim=metadata.shape[1], num_classes=len(label_to_idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = {"ce": 0.5, "ssl": 1.0, "disc": 0.1, "gen": 0.1, "q": 0.5, "club": 0.01, "rec": 0.1}

    best = {"auc": -1.0}
    out_dir = Path(args.output_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        train_stats, _ = run_epoch(model, train_loader, optimizer, device, weights, debug_shapes=args.debug_shapes and epoch == 0)
        val_stats, val_outputs = run_epoch(model, val_loader, None, device, weights, debug_shapes=False, collect_outputs=True)
        row = {"epoch": epoch + 1, "train": train_stats, "val": val_stats}
        print(json.dumps(row, ensure_ascii=False))
        score = val_stats.get("auc", val_stats["acc"])
        if score > best.get("auc", -1.0):
            best = val_stats
            torch.save({"model": model.state_dict(), "args": vars(args), "labels": label_to_idx, "metrics": best}, out_dir / "best.pt")
            val_outputs.to_csv(out_dir / "val_predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
