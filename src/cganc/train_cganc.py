import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from cganc.data import CGANCDataset, collate, read_manifest
from cganc.model import CGANC, Discriminator3D, LatentClassifier


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
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "auc": roc_auc_score(y_true, prob[:, 1]) if len(np.unique(y_true)) == 2 else float("nan"),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        })
    else:
        result["pre"] = precision_score(y_true, pred, average="macro", zero_division=0)
        result["recall"] = recall_score(y_true, pred, average="macro", zero_division=0)
        result["f1"] = f1_score(y_true, pred, average="macro", zero_division=0)
        result["auc"] = roc_auc_score(y_true, prob, multi_class="ovr") if len(np.unique(y_true)) > 2 else float("nan")
    return result


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def train_gan_epoch(generator, disc_mri, disc_pet, loader, opt_g, opt_d, device, args, debug_shapes=False):
    generator.train()
    disc_mri.train()
    disc_pet.train()
    bce = torch.nn.BCEWithLogitsLoss()
    total_g, total_d = 0.0, 0.0
    debug_once = debug_shapes
    iterator = tqdm(loader, leave=False, desc="gan")
    for batch in iterator:
        mri = batch["mri"].to(device)
        pet = batch["pet"].to(device)
        real = torch.ones(mri.size(0), device=device)
        fake = torch.zeros(mri.size(0), device=device)

        set_requires_grad(disc_mri, True)
        set_requires_grad(disc_pet, True)
        for _ in range(args.critic_steps):
            with torch.no_grad():
                out = generator(mri, pet, debug_shapes=debug_once)
                debug_once = False
            d_loss = (
                bce(disc_mri(mri), real)
                + bce(disc_mri(out["rec_mri"].detach()), fake)
                + bce(disc_pet(pet), real)
                + bce(disc_pet(out["rec_pet"].detach()), fake)
            )
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

        set_requires_grad(disc_mri, False)
        set_requires_grad(disc_pet, False)
        g_loss = None
        for _ in range(args.generator_steps):
            out = generator(mri, pet)
            recon = F.mse_loss(out["rec_mri"], mri) + F.mse_loss(out["rec_pet"], pet)
            adv = bce(disc_mri(out["rec_mri"]), real) + bce(disc_pet(out["rec_pet"]), real)
            g_loss = args.alpha * recon + (1.0 - args.alpha) * adv
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()
        total_d += float(d_loss.item()) * mri.size(0)
        total_g += float(g_loss.item()) * mri.size(0)
        iterator.set_postfix(g=float(g_loss.item()), d=float(d_loss.item()))
    return {"g_loss": total_g / len(loader.dataset), "d_loss": total_d / len(loader.dataset)}


def run_classifier_epoch(generator, classifier, loader, optimizer, device, collect_outputs=False, debug_shapes=False):
    train = optimizer is not None
    generator.eval()
    classifier.train(train)
    all_y, all_prob, all_id, all_feat = [], [], [], []
    total_loss = 0.0
    debug_once = debug_shapes
    iterator = tqdm(loader, leave=False, desc="clf_train" if train else "clf_eval")
    for batch in iterator:
        mri = batch["mri"].to(device)
        pet = batch["pet"].to(device)
        target = batch["target"].to(device)
        with torch.no_grad():
            fused, _, _ = generator.encode(mri, pet)
        with torch.set_grad_enabled(train):
            logits, feat = classifier(fused.detach(), debug_shapes=debug_once)
            debug_once = False
            loss = F.cross_entropy(logits, target)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
        all_prob.append(prob)
        all_y.append(target.detach().cpu().numpy())
        if collect_outputs:
            all_feat.append(feat.detach().cpu().numpy())
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


def train_fold(args, frame, train_idx, val_idx, fold, label_to_idx):
    train_ds = CGANCDataset(frame.iloc[train_idx], args.input_shape, augment=True)
    val_ds = CGANCDataset(frame.iloc[val_idx], args.input_shape, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    generator = CGANC(args.latent_channels, args.base_channels).to(device)
    disc_mri = Discriminator3D(args.base_channels).to(device)
    disc_pet = Discriminator3D(args.base_channels).to(device)
    classifier = LatentClassifier(args.latent_channels, len(label_to_idx)).to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(list(disc_mri.parameters()) + list(disc_pet.parameters()), lr=args.gan_lr, betas=(0.5, 0.999))
    opt_c = torch.optim.Adam(classifier.parameters(), lr=args.clf_lr, weight_decay=args.weight_decay)

    out_dir = Path(args.output_dir) / args.task / f"fold_{fold:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.gan_epochs + 1):
        gan_stats = train_gan_epoch(generator, disc_mri, disc_pet, train_loader, opt_g, opt_d, device, args, args.debug_shapes and epoch == 1)
        print(json.dumps({"phase": "gan", "epoch": epoch, **gan_stats}, ensure_ascii=False))

    best = {"bal_acc": -1.0}
    for epoch in range(1, args.clf_epochs + 1):
        train_stats, _ = run_classifier_epoch(generator, classifier, train_loader, opt_c, device, debug_shapes=args.debug_shapes and epoch == 1)
        val_stats, val_outputs = run_classifier_epoch(generator, classifier, val_loader, None, device, collect_outputs=True)
        row = {"phase": "classifier", "epoch": epoch, "train": train_stats, "val": val_stats}
        print(json.dumps(row, ensure_ascii=False))
        score = val_stats.get("bal_acc", val_stats["acc"])
        if score > best.get("bal_acc", -1.0):
            best = val_stats
            torch.save({
                "generator": generator.state_dict(),
                "classifier": classifier.state_dict(),
                "args": vars(args),
                "labels": label_to_idx,
                "metrics": best,
            }, out_dir / "best.pt")
            val_outputs.to_csv(out_dir / "val_predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    return best


def main():
    parser = argparse.ArgumentParser(description="Reproduce coupled-GAN MRI/PET fusion classifier on local ADNI data.")
    parser.add_argument("--data-root", default=r"D:\AD\心睿\ADNI-amyloid-smri-pet")
    parser.add_argument("--mri-folder", default="mwp1")
    parser.add_argument("--pet-folder", default="pet")
    parser.add_argument("--task", default="AD_CN", choices=["AD_CN", "MCI_CN", "AD_MCI", "CN_MCI_AD"])
    parser.add_argument("--output-dir", default="cganc_runs")
    parser.add_argument("--input-shape", type=parse_shape, default=(64, 80, 64))
    parser.add_argument("--gan-epochs", type=int, default=20)
    parser.add_argument("--clf-epochs", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--single-split", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--latent-channels", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--critic-steps", type=int, default=1)
    parser.add_argument("--generator-steps", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gan-lr", type=float, default=2e-4)
    parser.add_argument("--clf-lr", type=float, default=1e-4)
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
