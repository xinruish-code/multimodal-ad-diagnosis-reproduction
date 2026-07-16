from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from ppbhg.nifti import load_nifti


TASKS = {
    "AD_CN": ["CN", "AD"],
    "MCI_CN": ["CN", "MCI"],
    "AD_MCI": ["MCI", "AD"],
    "CN_MCI_AD": ["CN", "MCI", "AD"],
}


def read_manifest(data_root, task, brain_pet_folder="pet", heart_folder="mwp1", gut_folder="wm", mri_folder="mwp1"):
    if task not in TASKS:
        raise ValueError(f"task must be one of {sorted(TASKS)}")
    data_root = Path(data_root)
    frame = pd.read_csv(data_root / "ADNI_amyloid_smri_pet.csv")
    labels = TASKS[task]
    frame = frame[frame["Label"].isin(labels)].copy()
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    frame["target"] = frame["Label"].map(label_to_idx).astype(int)
    frame["brain_pet_path"] = frame["ID"].map(lambda name: str(data_root / brain_pet_folder / name))
    frame["heart_path"] = frame["ID"].map(lambda name: str(data_root / heart_folder / name))
    frame["gut_path"] = frame["ID"].map(lambda name: str(data_root / gut_folder / name))
    frame["mri_path"] = frame["ID"].map(lambda name: str(data_root / mri_folder / name))
    for col in ["brain_pet_path", "heart_path", "gut_path", "mri_path"]:
        frame = frame[frame[col].map(lambda p: Path(p).exists())]
    return frame.reset_index(drop=True), label_to_idx


class PPBHGDataset(Dataset):
    def __init__(self, frame, input_shape=(64, 80, 64), student_modality="pet", augment=False):
        self.frame = frame.reset_index(drop=True)
        self.input_shape = tuple(input_shape)
        self.student_modality = student_modality
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
        heart = self._load_volume(row.heart_path)
        gut = self._load_volume(row.gut_path)
        mri = self._load_volume(row.mri_path)
        if self.augment and np.random.rand() < 0.5:
            brain_pet = brain_pet[:, :, :, ::-1].copy()
            heart = heart[:, :, :, ::-1].copy()
            gut = gut[:, :, :, ::-1].copy()
            mri = mri[:, :, :, ::-1].copy()
        if self.student_modality == "pet":
            student_image = brain_pet
        elif self.student_modality == "mri":
            student_image = mri
        elif self.student_modality == "pet_mri":
            student_image = np.concatenate([brain_pet, mri], axis=0)
        else:
            raise ValueError("student_modality must be pet, mri, or pet_mri")
        return {
            "brain_pet": torch.from_numpy(brain_pet),
            "heart": torch.from_numpy(heart),
            "gut": torch.from_numpy(gut),
            "mri": torch.from_numpy(mri),
            "student_image": torch.from_numpy(student_image),
            "target": torch.tensor(int(row.target), dtype=torch.long),
            "id": row.ID,
        }


def collate(batch):
    keys = ["brain_pet", "heart", "gut", "mri", "student_image", "target"]
    out = {key: torch.stack([item[key] for item in batch]) for key in keys}
    out["id"] = [item["id"] for item in batch]
    return out

