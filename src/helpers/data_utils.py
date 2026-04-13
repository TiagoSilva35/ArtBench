from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from torchvision import transforms as T

from src.helpers.csv_handler import load_ids_from_training_csv


def build_image_transform(image_size: int):
    return T.Compose(
        [
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def unpack_images(batch):
    return batch[0]


def resolve_train_indices(
    train_hf,
    stage: str,
    training_csv_path: Path | str,
    index_column: str,
    debug_fn: Callable[[Any], None] | None = None,
) -> list[int]:
    if stage == "final100":
        train_indices = list(range(len(train_hf)))
        if debug_fn is not None:
            debug_fn(f"Using full train split: {len(train_indices)} samples (stage={stage})")
        return train_indices

    train_ids_from_csv = load_ids_from_training_csv(Path(training_csv_path), index_column=index_column)
    max_idx = len(train_hf) - 1
    train_indices = [i for i in train_ids_from_csv if 0 <= i <= max_idx]

    if debug_fn is not None:
        debug_fn(f"Loaded ids: {len(train_ids_from_csv)}")
        debug_fn(f"First 10 ids: {train_ids_from_csv[:10]}")
        if len(train_indices) != len(train_ids_from_csv):
            debug_fn(
                f"Filtered {len(train_ids_from_csv) - len(train_indices)} ids outside [0, {max_idx}]"
            )
        debug_fn(f"Using CSV subset size: {len(train_indices)} (stage={stage})")

    return train_indices
