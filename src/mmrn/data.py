from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from mdl_net.nifti import load_nifti


TASKS = {
    "AD_CN": ["CN", "AD"],
    "MCI_CN": ["CN", "MCI"],
    "AD_MCI": ["MCI", "AD"],
    "CN_MCI_AD": ["CN", "MCI", "AD"],
}


DEFAULT_META_COLUMNS = ["Age", "Sex", "MMSE Total Score"]


def read_manifest(data_root, task, image_folder="mwp1", meta_columns=None):
    if task not in TASKS:
        raise ValueError(f"task must be one of {sorted(TASKS)}")
    data_root = Path(data_root)
    df = pd.read_csv(data_root / "ADNI_amyloid_smri_pet.csv")
    labels = TASKS[task]
    df = df[df["Label"].isin(labels)].copy()
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    df["target"] = df["Label"].map(label_to_idx).astype(int)
    df["image_path"] = df["ID"].map(lambda name: str(data_root / image_folder / name))
    df = df[df["image_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)

    meta_columns = meta_columns or DEFAULT_META_COLUMNS
    meta = build_metadata(df, meta_columns)
    return df, meta, label_to_idx


def build_metadata(df, columns):
    values = []
    names = []
    for col in columns:
        if col not in df.columns:
            continue
        if col == "Sex":
            sex = df[col].map({"F": 0.0, "M": 1.0}).astype("float32")
            values.append(sex.to_numpy()[:, None])
            names.append("Sex")
        else:
            numeric = pd.to_numeric(df[col], errors="coerce").astype("float32")
            numeric = numeric.fillna(float(numeric.mean()))
            std = float(numeric.std()) or 1.0
            values.append(((numeric - float(numeric.mean())) / std).to_numpy()[:, None])
            names.append(col)
    if not values:
        raise ValueError("No usable metadata columns found")
    meta = np.concatenate(values, axis=1).astype("float32")
    return pd.DataFrame(meta, columns=names, index=df.index)


class MMRNDataset(Dataset):
    def __init__(self, frame, metadata, input_shape=(64, 80, 64), augment=True):
        self.frame = frame.reset_index(drop=True)
        self.metadata = metadata.reset_index(drop=True)
        self.input_shape = tuple(input_shape)
        self.augment = augment

    def __len__(self):
        return len(self.frame)

    def _load_volume(self, path):
        vol = load_nifti(path)
        vol = zoom(vol, [n / o for n, o in zip(self.input_shape, vol.shape)], order=1)
        vol = np.where(np.isfinite(vol), vol, 0.0)
        vol = (vol - float(vol.mean())) / (float(vol.std()) + 1e-6)
        return torch.from_numpy(vol[None].astype("float32"))

    def _view(self, x):
        if not self.augment:
            return x
        y = x
        if torch.rand(()) < 0.5:
            y = torch.flip(y, dims=[-1])
        scale = 0.9 + 0.2 * torch.rand(())
        y = y * scale + 0.03 * torch.randn_like(y)
        if torch.rand(()) < 0.5:
            shift = int(torch.randint(-2, 3, ()).item())
            y = torch.roll(y, shifts=shift, dims=-2)
        return y

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        x = self._load_volume(row.image_path)
        return {
            "view_i": self._view(x.clone()),
            "view_j": self._view(x.clone()),
            "target": torch.tensor(int(row.target), dtype=torch.long),
            "meta": torch.from_numpy(self.metadata.iloc[idx].to_numpy(dtype="float32").copy()),
            "id": row.ID,
        }


def collate(batch):
    return {
        "view_i": torch.stack([b["view_i"] for b in batch]),
        "view_j": torch.stack([b["view_j"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "meta": torch.stack([b["meta"] for b in batch]),
        "id": [b["id"] for b in batch],
    }
