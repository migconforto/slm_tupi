"""Reproduces the split of notebooks/sft.ipynb and keeps its ids in splits/ (R4.7, P6)."""
import hashlib
import json
import os

import pandas as pd
from datasets import Dataset

from tupi_exp import SPLITS_DIR
from tupi_exp.config import SPLIT_SEED, TRAIN_SHARE, V2_PATH
from utils.utils_train import pre_process


def reproduce_split(dataset_path=V2_PATH, seed=SPLIT_SEED):
    """Same steps as cell 75c796c5 of notebooks/sft.ipynb. Returns (train_ids, val_ids)."""
    dataset = Dataset.from_pandas(pre_process(pd.read_json(dataset_path))).shuffle(seed=seed)
    n_train = int(TRAIN_SHARE * len(dataset))
    ids = dataset["id"]
    return ids[:n_train], ids[n_train:]


def ids_hash(ids):
    """SHA-256 of the sorted id list, so the order of the examples does not matter."""
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def split_path(name, seed=SPLIT_SEED):
    return os.path.join(SPLITS_DIR, f"seed{seed}_{name}_ids.json")


def save_split(train_ids, val_ids, seed=SPLIT_SEED):
    os.makedirs(SPLITS_DIR, exist_ok=True)
    for name, ids in (("train", train_ids), ("val", val_ids)):
        with open(split_path(name, seed), "w", encoding="utf-8", newline="\n") as f:
            json.dump(ids, f, indent=0)


def load_split(seed=SPLIT_SEED):
    ids = []
    for name in ("train", "val"):
        with open(split_path(name, seed), encoding="utf-8") as f:
            ids.append(json.load(f))
    return tuple(ids)
