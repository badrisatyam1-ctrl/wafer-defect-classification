"""
Dataset and split utilities for full-wafer classification.

Design decisions:
- The dataset never crops or tiles wafers; each sample stays full-frame.
- Holdout is done by lot or time, not random sample shuffling.
- Training augmentation is limited to geometry-preserving transforms.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.resnet18_classifier import WaferPreprocessor, get_train_augmentations


@dataclass
class WaferDatasetBundle:
    """In-memory wafer dataset with metadata used for leakage-safe splitting."""

    images: np.ndarray
    labels: np.ndarray
    lot_ids: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None

    def __post_init__(self):
        self.images = np.asarray(self.images)
        self.labels = np.asarray(self.labels, dtype=np.int64)
        if self.lot_ids is not None:
            self.lot_ids = np.asarray(self.lot_ids)
        if self.timestamps is not None:
            self.timestamps = np.asarray(self.timestamps)

        if len(self.images) != len(self.labels):
            raise ValueError("images and labels must have the same length.")
        if self.lot_ids is not None and len(self.lot_ids) != len(self.labels):
            raise ValueError("lot_ids must align 1:1 with labels.")
        if self.timestamps is not None and len(self.timestamps) != len(self.labels):
            raise ValueError("timestamps must align 1:1 with labels.")

    def subset(self, indices: Sequence[int]) -> "WaferDatasetBundle":
        indices = np.asarray(indices, dtype=np.int64)
        return WaferDatasetBundle(
            images=self.images[indices],
            labels=self.labels[indices],
            lot_ids=None if self.lot_ids is None else self.lot_ids[indices],
            timestamps=None if self.timestamps is None else self.timestamps[indices],
        )


def _group_split_indices(
    groups: np.ndarray,
    val_fraction: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = random.Random(seed)
    unique_groups = list(dict.fromkeys(groups.tolist()))
    rng.shuffle(unique_groups)

    n_val_groups = max(1, int(round(len(unique_groups) * val_fraction)))
    val_groups = set(unique_groups[:n_val_groups])

    train_idx = [idx for idx, group in enumerate(groups) if group not in val_groups]
    val_idx = [idx for idx, group in enumerate(groups) if group in val_groups]

    return {
        "train": np.asarray(train_idx, dtype=np.int64),
        "val": np.asarray(val_idx, dtype=np.int64),
    }


def split_dataset_bundle(
    bundle: WaferDatasetBundle,
    split_mode: str = "lot",
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Dict[str, WaferDatasetBundle]:
    """
    Split by lot or time, never randomly by wafer.

    `split_mode="lot"` is the default because it protects against lot-level
    process leakage. `split_mode="time"` holds out the newest wafers.
    """

    split_mode = split_mode.lower()
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")

    if split_mode == "lot":
        if bundle.lot_ids is None:
            raise ValueError("lot_ids are required when split_mode='lot'.")
        indices = _group_split_indices(bundle.lot_ids, val_fraction=val_fraction, seed=seed)
    elif split_mode == "time":
        if bundle.timestamps is None:
            raise ValueError("timestamps are required when split_mode='time'.")
        order = np.argsort(bundle.timestamps, kind="stable")
        n_val = max(1, int(round(len(order) * val_fraction)))
        val_idx = order[-n_val:]
        train_idx = order[:-n_val]
        indices = {
            "train": np.asarray(train_idx, dtype=np.int64),
            "val": np.asarray(val_idx, dtype=np.int64),
        }
    else:
        raise ValueError("split_mode must be 'lot' or 'time'.")

    return {
        "train": bundle.subset(indices["train"]),
        "val": bundle.subset(indices["val"]),
    }


class WaferMapDataset(Dataset):
    """
    PyTorch dataset for full-wafer classification.

    Training augmentation is applied before the same shared preprocessor that
    inference uses, which keeps preprocessing parity explicit.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        preprocessor: Optional[WaferPreprocessor] = None,
        training: bool = False,
    ):
        self.images = np.asarray(images)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.preprocessor = preprocessor or WaferPreprocessor()
        self.training = training
        self.augment = get_train_augmentations() if training else None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = self.images[index]
        label = int(self.labels[index])

        prepared = self.preprocessor.prepare_image(image)
        if self.training and self.augment is not None:
            prepared = self.augment(prepared)

        tensor = self.preprocessor.tensorize(prepared)
        return tensor, label


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency weights and normalise them to mean 1.0.

    This keeps the loss scale stable while still up-weighting minority classes.
    """

    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def compute_sample_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Compute per-sample weights for a WeightedRandomSampler.

    This complements class-weighted loss by increasing the chance that rare
    macro defect classes appear in each mini-batch.
    """

    class_weights = compute_class_weights(labels, num_classes=num_classes).cpu().numpy()
    sample_weights = class_weights[np.asarray(labels, dtype=np.int64)]
    return torch.tensor(sample_weights, dtype=torch.double)


if __name__ == "__main__":
    images = (np.random.rand(32, 240, 200, 3) * 255).astype(np.uint8)
    labels = np.random.randint(0, 8, size=32)
    lots = np.array([f"LOT_{idx // 4:03d}" for idx in range(32)])
    times = np.arange(32)
    bundle = WaferDatasetBundle(images=images, labels=labels, lot_ids=lots, timestamps=times)
    split = split_dataset_bundle(bundle, split_mode="lot", val_fraction=0.25)
    dataset = WaferMapDataset(split["train"].images, split["train"].labels, training=True)
    tensor, label = dataset[0]
    print("train_shape:", tuple(tensor.shape))
    print("label:", label)
