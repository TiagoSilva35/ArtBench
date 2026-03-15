from __future__ import annotations

import sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pathlib import Path
from collections import Counter
from scripts.artbench_local_dataset import load_kaggle_artbench10_splits
from src.config import SEED, PROJECT_ROOT, KAGGLE_ROOT, SCRIPTS_DIR, IMAGE_SIZE, TRAIN_FRACTION, BATCH_SIZE, TRAINING_CSV_PATH, INDEX_COLUMN
from src.helpers.debugger import DBG
from src.dataset_manager.HFloader import HFDatasetTorch
from src.helpers.utils import make_subset_indices, show_batch_grid
from src.helpers.csv_handler import load_ids_from_training_csv, export_split_to_folder

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DBG(f'PROJECT_ROOT = {PROJECT_ROOT}')
DBG(f'KAGGLE_ROOT  = {KAGGLE_ROOT}')

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


if __name__ == '__main__':

    hf_ds = load_kaggle_artbench10_splits(KAGGLE_ROOT)
    train_hf = hf_ds["train"]

    DBG(f"Train size: {len(train_hf)}")
    DBG(f"Columns   : {train_hf.column_names}")

    label_feature = train_hf.features["label"]
    class_names = list(label_feature.names)
    num_classes = len(class_names)
    DBG(f"Num classes: {num_classes}")
    DBG(f"Class names: {class_names}")

    # Class distribution summary
    train_counts = Counter(train_hf["label"])

    DBG("\nTrain class distribution:")
    for cid, name in enumerate(class_names):
        DBG(f"  {cid:2d} | {name:>15s} | {train_counts.get(cid, 0):6d}")

    transform = T.Compose([
        T.Resize(IMAGE_SIZE, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),  
    ])

    train_indices = make_subset_indices(len(train_hf), TRAIN_FRACTION, seed=SEED)

    train_ds = HFDatasetTorch(train_hf, transform=transform, indices=train_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    DBG(f"Train dataset length (after fraction): {len(train_ds)}")
    DBG(f"Train batches                        : {len(train_loader)}")


    train_ids_from_csv = load_ids_from_training_csv(TRAINING_CSV_PATH, index_column=INDEX_COLUMN)
    DBG(f'Loaded ids: {len(train_ids_from_csv)}')
    DBG(f'First 10 ids: {train_ids_from_csv[:10]}')

    train_ds_from_csv = HFDatasetTorch(train_hf, transform=transform, indices=train_ids_from_csv)
    train_loader_from_csv = DataLoader(
        train_ds_from_csv,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    DBG(f'Subset train dataset length: {len(train_ds_from_csv)}')
    DBG(f'Subset train batches      : {len(train_loader_from_csv)}')
    show_batch_grid(train_loader, class_names, n_images=36, nrow=6, title='ArtBench-10 Train Samples')

    EXPORT_ROOT = Path('exported_data')
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

    export_split_to_folder(train_loader, class_names, EXPORT_ROOT / 'train_subset', max_images=500)
