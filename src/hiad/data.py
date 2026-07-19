from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from hiad.nifti import load_nifti


TASKS = {
    "AD_CN": ["CN", "AD"],
    "MCI_CN": ["CN", "MCI"],
    "AD_MCI": ["MCI", "AD"],
    "CN_MCI_AD": ["CN", "MCI", "AD"],
}


DEFAULT_CAD_COLUMNS = ["Age", "Sex", "MMSE Total Score"]


def _encode_cad(frame, columns):
    pieces = []
    used = []
    for col in columns:
        if col not in frame.columns:
            continue
        series = frame[col]
        if series.dtype == object:
            dummies = pd.get_dummies(series.fillna("UNK").astype(str), prefix=col)
            pieces.append(dummies.astype("float32"))
            used.extend(dummies.columns.tolist())
        else:
            values = pd.to_numeric(series, errors="coerce")
            values = values.fillna(values.median() if not pd.isna(values.median()) else 0.0)
            std = float(values.std()) if not pd.isna(values.std()) and float(values.std()) > 1e-6 else 1.0
            normalized = ((values - float(values.mean())) / std).replace([np.inf, -np.inf], 0.0).fillna(0.0)
            pieces.append(pd.DataFrame({col: normalized.astype("float32")}))
            used.append(col)
    if not pieces:
        pieces = [pd.DataFrame({"cad_bias": np.zeros(len(frame), dtype="float32")})]
        used = ["cad_bias"]
    cad = pd.concat(pieces, axis=1).astype("float32")
    return cad, used


def read_manifest(data_root, task, mri_folder="mwp1", pet_folder="pet", cad_columns=None):
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
    cad, used_cad = _encode_cad(frame, cad_columns or DEFAULT_CAD_COLUMNS)
    for col in cad.columns:
        frame[f"cad__{col}"] = cad[col].to_numpy(dtype="float32")
    return frame, label_to_idx, used_cad


class HIADDataset(Dataset):
    def __init__(self, frame, input_shape=(64, 80, 64), missing_rate=0.0, missing_eval=False, seed=42, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.input_shape = tuple(input_shape)
        self.missing_rate = float(missing_rate)
        self.missing_eval = missing_eval
        self.seed = int(seed)
        self.augment = augment
        self.cad_cols = [c for c in frame.columns if c.startswith("cad__")]

    def __len__(self):
        return len(self.frame)

    def _load_volume(self, path):
        vol = load_nifti(path)
        vol = zoom(vol, [n / o for n, o in zip(self.input_shape, vol.shape)], order=1)
        vol = np.where(np.isfinite(vol), vol, 0.0)
        vol = (vol - float(vol.mean())) / (float(vol.std()) + 1e-6)
        return vol[None].astype("float32")

    def _mask(self, idx):
        if self.missing_rate <= 0:
            return np.ones(3, dtype="float32")
        rng = np.random.default_rng(self.seed + idx + (100000 if self.missing_eval else 0))
        rho = (rng.random(3) >= self.missing_rate).astype("float32")
        if rho.sum() == 0:
            rho[int(rng.integers(0, 3))] = 1.0
        return rho

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        mri = self._load_volume(row.mri_path)
        pet = self._load_volume(row.pet_path)
        if self.augment and np.random.rand() < 0.5:
            mri = mri[:, :, :, ::-1].copy()
            pet = pet[:, :, :, ::-1].copy()
        cad = row[self.cad_cols].to_numpy(dtype="float32")
        rho = self._mask(idx)
        return {
            "mri": torch.from_numpy(mri),
            "pet": torch.from_numpy(pet),
            "cad": torch.from_numpy(cad),
            "rho": torch.from_numpy(rho),
            "target": torch.tensor(int(row.target), dtype=torch.long),
            "id": row.ID,
        }


def collate(batch):
    return {
        "mri": torch.stack([item["mri"] for item in batch]),
        "pet": torch.stack([item["pet"] for item in batch]),
        "cad": torch.stack([item["cad"] for item in batch]),
        "rho": torch.stack([item["rho"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "id": [item["id"] for item in batch],
    }

