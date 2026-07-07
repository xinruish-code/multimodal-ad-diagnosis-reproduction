from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from .nifti import load_nifti


TASKS = {
    "AD_CN": ("AD", "CN"),
    "MCI_CN": ("MCI", "CN"),
    "AD_MCI": ("AD", "MCI"),
}


def read_manifest(data_root, task):
    data_root = Path(data_root)
    csv_path = data_root / "ADNI_amyloid_smri_pet.csv"
    df = pd.read_csv(csv_path)
    if task not in TASKS:
        raise ValueError(f"task must be one of {sorted(TASKS)}")

    positive, negative = TASKS[task]
    df = df[df["Label"].isin([positive, negative])].copy()
    df["target"] = (df["Label"] == positive).astype(int)

    for col, folder in [("pet_path", "pet"), ("gm_path", "mwp1"), ("wm_path", "wm")]:
        df[col] = df["ID"].map(lambda name: str(data_root / folder / name))
        df = df[df[col].map(lambda p: Path(p).exists())]

    return df.reset_index(drop=True), {"negative": negative, "positive": positive}


def load_roi_table(path):
    if path is None:
        return None
    roi = pd.read_csv(path)
    key = "ID" if "ID" in roi.columns else roi.columns[0]
    roi_cols = [c for c in roi.columns if c.lower().startswith("roi")]
    if len(roi_cols) < 90:
        numeric_cols = [c for c in roi.columns if c != key and pd.api.types.is_numeric_dtype(roi[c])]
        roi_cols = numeric_cols[:90]
    if len(roi_cols) != 90:
        raise ValueError("ROI table must contain 90 ROI columns, e.g. roi_001..roi_090")
    return roi.set_index(key)[roi_cols].astype("float32")


class ADNIMultimodalDataset(Dataset):
    def __init__(self, frame, input_shape=(96, 112, 96), augment=False, roi_table=None):
        self.frame = frame.reset_index(drop=True)
        self.input_shape = tuple(input_shape)
        self.augment = augment
        self.roi_table = roi_table

    def __len__(self):
        return len(self.frame)

    def _load_volume(self, path):
        vol = load_nifti(path)
        factors = [n / o for n, o in zip(self.input_shape, vol.shape)]
        vol = zoom(vol, factors, order=1)
        finite = np.isfinite(vol)
        if not finite.all():
            vol = np.where(finite, vol, 0.0)
        mean = float(vol.mean())
        std = float(vol.std())
        vol = (vol - mean) / (std + 1e-6)
        return vol[None]

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        pet = self._load_volume(row.pet_path)
        gm = self._load_volume(row.gm_path)
        wm = self._load_volume(row.wm_path)
        image = np.concatenate([pet, gm, wm], axis=0)
        if self.augment and np.random.rand() < 0.5:
            image = image[:, :, :, ::-1].copy()

        sample = {
            "image": torch.from_numpy(image.astype("float32")),
            "target": torch.tensor(int(row.target), dtype=torch.long),
            "id": row.ID,
        }
        if self.roi_table is not None and row.ID in self.roi_table.index:
            roi = self.roi_table.loc[row.ID].to_numpy(dtype="float32")
            sample["roi"] = torch.from_numpy(roi)
        return sample
