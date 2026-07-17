from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from e2ad.nifti import load_nifti


TASKS = {
    "AD_CN": ["CN", "AD"],
    "MCI_CN": ["CN", "MCI"],
    "AD_MCI": ["MCI", "AD"],
    "CN_MCI_AD": ["CN", "MCI", "AD"],
}


def read_manifest(data_root, task, mri_folder="mwp1", pet_folder="pet"):
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
    frame = frame[frame["pet_path"].map(lambda p: Path(p).exists())]
    return frame.reset_index(drop=True), label_to_idx


class E2ADDataset(Dataset):
    def __init__(self, frame, input_shape=(64, 80, 64), augment=False):
        self.frame = frame.reset_index(drop=True)
        self.input_shape = tuple(input_shape)
        self.augment = augment

    def __len__(self):
        return len(self.frame)

    def _load_volume(self, path):
        vol = load_nifti(path)
        vol = zoom(vol, [n / o for n, o in zip(self.input_shape, vol.shape)], order=1)
        vol = np.where(np.isfinite(vol), vol, 0.0)
        lo, hi = np.percentile(vol, [1, 99])
        vol = np.clip(vol, lo, hi)
        vol = (vol - float(vol.mean())) / (float(vol.std()) + 1e-6)
        return vol[None].astype("float32")

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        mri = self._load_volume(row.mri_path)
        pet = self._load_volume(row.pet_path)
        if self.augment and np.random.rand() < 0.5:
            mri = mri[:, :, :, ::-1].copy()
            pet = pet[:, :, :, ::-1].copy()
        return {
            "mri": torch.from_numpy(mri),
            "pet": torch.from_numpy(pet),
            "target": torch.tensor(int(row.target), dtype=torch.long),
            "id": row.ID,
        }


def collate(batch):
    return {
        "mri": torch.stack([item["mri"] for item in batch]),
        "pet": torch.stack([item["pet"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "id": [item["id"] for item in batch],
    }

