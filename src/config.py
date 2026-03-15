import sys
import random
import numpy as np
import torch
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

if not KAGGLE_ROOT.exists() or not (SCRIPTS_DIR / 'artbench_local_dataset.py').exists():
    raise FileNotFoundError(
        'Could not resolve project folders from relative paths. '
        'Run this notebook from student_start_pack/ or adjust PROJECT_ROOT.'
    )
