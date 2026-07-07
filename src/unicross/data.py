from pathlib import Path

import numpy as np
import pandas as pd
import torch
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


def _encode_metadata(frame, columns):
    pieces = []
    used = []
    for col in columns:
        if col not in frame.columns:
            continue
        series = frame[col]
        if series.dtype == object:
            values = series.fillna("UNK").astype(str)
            dummies = pd.get_dummies(values, prefix=col)
            pieces.append(dummies.astype("float32"))
            used.extend(list(dummies.columns))
        else:
            values = pd.to_numeric(series, errors="coerce")
            median = values.median()
            if pd.isna(median):
                values = values.fillna(0.0)
            else:
                values = values.fillna(median)
            mean = float(values.mean()) if not pd.isna(values.mean()) else 0.0
            std_value = values.std()
            std = float(std_value) if not pd.isna(std_value) and float(std_value) > 1e-6 else 1.0
            normalized = ((values - mean) / std).replace([np.inf, -np.inf], 0.0).fillna(0.0)
            pieces.append(pd.DataFrame({col: normalized.astype("float32")}))
            used.append(col)
    if not pieces:
        pieces = [pd.DataFrame({"meta_bias": np.zeros(len(frame), dtype="float32")})]
        used = ["meta_bias"]
    meta = pd.concat(pieces, axis=1).replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float32")
    return meta, used


def read_manifest(data_root, task, mri_folder="mwp1", pet_folder="pet", meta_columns=None):
    if task not in TASKS:
        raise ValueError(f"task must be one of {sorted(TASKS)}")
    data_root = Path(data_root)
    frame = pd.read_csv(data_root / "ADNI_amyloid_smri_pet.csv")
    labels = TASKS[task]
    frame = frame[frame["Label"].isin(labels)].copy()
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    frame["target"] = frame["Label"].map(label_to_idx).astype(int)
    frame["mri_path"] = frame["ID"].map(lambda name: str(data_root / mri_folder / name))
    frame["pet_path"] = frame["ID"].map(lambda name: str(data_root / pet_folder / name))
    frame = frame[frame["mri_path"].map(lambda p: Path(p).exists())]
    frame = frame[frame["pet_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
    meta, used_meta = _encode_metadata(frame, meta_columns or DEFAULT_META_COLUMNS)
    for col in meta.columns:
        frame[f"meta__{col}"] = meta[col].to_numpy(dtype="float32")
    return frame, label_to_idx, used_meta


class UniCrossDataset(Dataset):
    def __init__(self, frame, input_shape=(64, 80, 64), augment=False):
        self.frame = frame.reset_index(drop=True)
        self.input_shape = tuple(input_shape)
        self.augment = augment
        self.meta_cols = [c for c in frame.columns if c.startswith("meta__")]

    def __len__(self):
        return len(self.frame)

    def _load_volume(self, path):
        vol = load_nifti(path)
        vol = zoom(vol, [n / o for n, o in zip(self.input_shape, vol.shape)], order=1)
        vol = np.where(np.isfinite(vol), vol, 0.0)
        vol = (vol - float(vol.mean())) / (float(vol.std()) + 1e-6)
        return vol[None].astype("float32")

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        mri = self._load_volume(row.mri_path)
        pet = self._load_volume(row.pet_path)
        if self.augment and np.random.rand() < 0.5:
            mri = mri[:, :, :, ::-1].copy()
            pet = pet[:, :, :, ::-1].copy()
        meta = row[self.meta_cols].to_numpy(dtype="float32")
        return {
            "mri": torch.from_numpy(mri),
            "pet": torch.from_numpy(pet),
            "meta": torch.from_numpy(meta),
            "target": torch.tensor(int(row.target), dtype=torch.long),
            "id": row.ID,
        }


def collate(batch):
    return {
        "mri": torch.stack([item["mri"] for item in batch]),
        "pet": torch.stack([item["pet"] for item in batch]),
        "meta": torch.stack([item["meta"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "id": [item["id"] for item in batch],
    }
