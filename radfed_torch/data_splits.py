"""Read the released train/validation/test client lists without resampling them.

The experiment configurations select an outer fold. Training and calibration
use its training clients; budget selection uses validation clients; final
evaluation uses test clients. An optional inner fold supplies a separate
validation set while retaining the outer test set.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ClientSplit:
    train_ids: np.ndarray
    validation_ids: np.ndarray
    test_ids: np.ndarray


def _load_ids(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", dtype=np.int64)
    return np.atleast_1d(values).astype(np.int64)


def load_client_split(
    directory: str | Path,
    fold: int,
    inner_fold: int | None = None,
    split_prefix: str = "fold",
) -> ClientSplit:
    """Load client IDs in their saved order, including the optional inner split."""
    root = Path(directory)
    split_prefix = str(split_prefix)
    if (
        not split_prefix
        or Path(split_prefix).name != split_prefix
        or any(separator in split_prefix for separator in ("/", "\\"))
    ):
        raise ValueError("split_prefix must be a nonempty filename prefix")
    train_ids = _load_ids(root / f"{split_prefix}{fold}_tr_client_ids.lst")
    validation_ids = _load_ids(root / f"{split_prefix}{fold}_val_client_ids.lst")
    test_ids = _load_ids(root / f"{split_prefix}{fold}_te_client_ids.lst")
    if inner_fold is not None:
        if int(inner_fold) == int(fold):
            raise ValueError("inner_fold must differ from the held-out outer fold")
        candidate_train_ids = np.unique(np.concatenate([train_ids, validation_ids]))
        inner_validation_ids = _load_ids(root / f"fold{int(inner_fold)}_te_client_ids.lst")
        if not np.all(np.isin(inner_validation_ids, candidate_train_ids)):
            raise ValueError("inner validation fold overlaps the outer test fold")
        train_ids = np.setdiff1d(candidate_train_ids, inner_validation_ids)
        validation_ids = inner_validation_ids
    return ClientSplit(train_ids, validation_ids, test_ids)
