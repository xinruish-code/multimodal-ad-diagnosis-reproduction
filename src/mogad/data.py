from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from mogad.nifti import load_nifti


TASKS = {
    "AD_CN": ["CN", "AD"],
    "MCI_CN": ["CN", "MCI"],
    "AD_MCI": ["MCI", "AD"],
    "CN_MCI_AD": ["CN", "MCI", "AD"],
}


def read_manifest(data_root, task, brain_pet_folder="pet", aux1_folder="mwp1", aux2_folder="wm"):
    if task not in TASKS:
        raise ValueError(f"task must be one of {sorted(TASKS)}")
    data_root = Path(data_root)
    frame = pd.read_csv(data_root / "ADNI_amyloid_smri_pet.csv")
    labels = TASKS[task]
    frame = frame[frame["Label"].isin(labels)].copy()
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    frame["target"] = frame["Label"].map(label_to_idx).astype(int)
    frame["brain_pet_path"] = frame["ID"].map(lambda name: str(data_root / brain_pet_folder / name))
    frame["aux1_path"] = frame["ID"].map(lambda name: str(data_root / aux1_folder / name))
    frame["aux2_path"] = frame["ID"].map(lambda name: str(data_root / aux2_folder / name))
    for col in ["brain_pet_path", "aux1_path", "aux2_path"]:
        frame = frame[frame[col].map(lambda p: Path(p).exists())]
    return frame.reset_index(drop=True), label_to_idx


class MOGADDataset(Dataset):
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
        vol = (vol - float(vol.mean())) / (float(vol.std()) + 1e-6)
        return vol[None].astype("float32")

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        brain_pet = self._load_volume(row.brain_pet_path)
        aux1 = self._load_volume(row.aux1_path)
        aux2 = self._load_volume(row.aux2_path)
        if self.augment and np.random.rand() < 0.5:
            brain_pet = brain_pet[:, :, :, ::-1].copy()
            aux1 = aux1[:, :, :, ::-1].copy()
            aux2 = aux2[:, :, :, ::-1].copy()
        return {
            "brain_pet": torch.from_numpy(brain_pet),
            "aux1": torch.from_numpy(aux1),
            "aux2": torch.from_numpy(aux2),
            "target": torch.tensor(int(row.target), dtype=torch.long),
            "id": row.ID,
        }


def collate(batch):
    return {
        "brain_pet": torch.stack([item["brain_pet"] for item in batch]),
        "aux1": torch.stack([item["aux1"] for item in batch]),
        "aux2": torch.stack([item["aux2"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "id": [item["id"] for item in batch],
    }
