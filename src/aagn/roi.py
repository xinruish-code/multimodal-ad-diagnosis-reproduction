from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import zoom

from mdl_net.nifti import load_nifti


def grid_roi_masks(shape, grid=(4, 4, 4)):
    d, h, w = shape
    gz, gy, gx = grid
    masks = []
    for zi in range(gz):
        z0, z1 = round(zi * d / gz), round((zi + 1) * d / gz)
        for yi in range(gy):
            y0, y1 = round(yi * h / gy), round((yi + 1) * h / gy)
            for xi in range(gx):
                x0, x1 = round(xi * w / gx), round((xi + 1) * w / gx)
                mask = np.zeros(shape, dtype="float32")
                mask[z0:z1, y0:y1, x0:x1] = 1.0
                masks.append(mask)
    return np.stack(masks, axis=0)


def load_atlas_masks(atlas_path, shape):
    atlas = load_nifti(atlas_path)
    atlas = zoom(atlas, [n / o for n, o in zip(shape, atlas.shape)], order=0)
    labels = [v for v in np.unique(atlas.astype("int32")) if v > 0]
    if not labels:
        raise ValueError(f"No positive ROI labels found in {atlas_path}")
    masks = [(atlas == label).astype("float32") for label in labels]
    return np.stack(masks, axis=0)


def build_roi_masks(input_shape, atlas_path=None, grid=(4, 4, 4)):
    if atlas_path and Path(atlas_path).exists():
        masks = load_atlas_masks(atlas_path, input_shape)
    else:
        masks = grid_roi_masks(input_shape, grid)
    masks = torch.from_numpy(masks.astype("float32"))
    return masks
