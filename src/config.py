import sys
import random
import numpy as np
import torch
import os
from pathlib import Path


SEED = 42
PROJECT_ROOT = Path()
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
KAGGLE_ROOT = PROJECT_ROOT / 'artbench-10-python'
IMAGE_SIZE = 32
BATCH_SIZE = 64
NUM_WORKERS = 2
TRAIN_FRACTION = 1.0  
TRAINING_CSV_PATH = Path('training_20_percent.csv')
INDEX_COLUMN = 'train_id_original' 

# Training workflow stage:
# - dev20: use CSV subset for fast iteration and HPO.
# - final100: use full train split for final model training/evaluation.
EXPERIMENT_STAGE = os.getenv('AB_EXPERIMENT_STAGE', 'dev20').strip().lower()
VALID_EXPERIMENT_STAGES = {'dev20', 'final100'}

if EXPERIMENT_STAGE not in VALID_EXPERIMENT_STAGES:
    raise ValueError(
        f"Invalid AB_EXPERIMENT_STAGE={EXPERIMENT_STAGE!r}. "
        f"Expected one of: {sorted(VALID_EXPERIMENT_STAGES)}"
    )

if not KAGGLE_ROOT.exists() or not (SCRIPTS_DIR / 'artbench_local_dataset.py').exists():
    raise FileNotFoundError(
        'Could not resolve project folders from relative paths. '
        'Run this notebook from student_start_pack/ or adjust PROJECT_ROOT.'
    )
