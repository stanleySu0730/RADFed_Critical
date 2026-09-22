from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
import pickle

import numpy as np
import torch
import torch.nn.functional as torch_functional

from language_utils import ALL_LETTERS


CHARACTER_MODEL_NAMES = {"lstm", "shakespeare", "transformer", "character-transformer"}


def _load_array(path: Path) -> np.ndarray:
    try:
        with path.open("rb") as stream:
            value = pickle.load(stream)
        return np.asarray(value)
    except (pickle.UnpicklingError, EOFError, ValueError, TypeError, AttributeError):
        return np.asarray(np.loadtxt(path, delimiter=",", dtype=np.float32))


def _load_ids(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", dtype=np.int64)
    return np.atleast_1d(values).astype(np.int64)


def _label_indices(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim > 1 and labels.shape[-1] > 1:
        return np.argmax(labels, axis=-1).astype(np.int64)
    if labels.dtype.kind in {"U", "S", "O"}:
        return np.asarray(
            [max(0, ALL_LETTERS.find(str(label)[0])) for label in labels.reshape(-1)],
            dtype=np.int64,
        )
    return labels.reshape(-1).astype(np.int64)


def _encode_sequences(values: np.ndarray, sequence_length: int) -> np.ndarray:
    encoded: list[list[int]] = []
    for raw in np.asarray(values):
        if isinstance(raw, np.ndarray):
            text = "".join(str(item) for item in raw.tolist())
        else:
            text = str(raw)
        indices = [max(0, ALL_LETTERS.find(character)) for character in text[:sequence_length]]
        indices.extend([0] * (sequence_length - len(indices)))
        encoded.append(indices)
    return np.asarray(encoded, dtype=np.int64)


@dataclass
class ClientDataset:
    features: np.ndarray
    labels: np.ndarray
    model_name: str
    sequence_length: int = 80
    _order: np.ndarray = field(init=False, repr=False)
    _position: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.features = np.asarray(self.features)
        if self.model_name.lower() in CHARACTER_MODEL_NAMES:
            self.features = _encode_sequences(self.features, self.sequence_length)
            raw_labels = np.asarray(self.labels).reshape(-1)
            self.labels = np.asarray(
                [max(0, ALL_LETTERS.find(str(value)[0])) for value in raw_labels],
                dtype=np.int64,
            )
        else:
            self.labels = _label_indices(self.labels)
        if len(self.features) != len(self.labels) or len(self.labels) == 0:
            raise ValueError("client features and labels must be nonempty and equally sized")
        self._order = np.arange(len(self.labels), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def next_indices(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        batch_size = len(self) if batch_size == -1 else int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive or -1")
        selected: list[np.ndarray] = []
        remaining = batch_size
        while remaining > 0:
            if self._position == 0:
                self._order = rng.permutation(len(self))
            available = len(self) - self._position
            take = min(remaining, available)
            selected.append(self._order[self._position:self._position + take])
            self._position = (self._position + take) % len(self)
            remaining -= take
        return np.concatenate(selected)

    def sample_indices(
        self, batch_size: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Draw an independent uniform mini-batch without advancing a cursor."""

        batch_size = len(self) if batch_size == -1 else int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive or -1")
        return rng.choice(
            len(self), size=batch_size, replace=batch_size > len(self)
        ).astype(np.int64)

    def tensors(
        self,
        indices: np.ndarray,
        device: torch.device,
        training: bool = False,
        rng: np.random.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features[indices]
        labels = torch.as_tensor(self.labels[indices], dtype=torch.long, device=device)
        name = self.model_name.lower()
        if name in CHARACTER_MODEL_NAMES:
            inputs = torch.as_tensor(features, dtype=torch.long, device=device)
        elif name in {"mobilenet", "mobilenetv2", "mbnt"}:
            inputs = torch.as_tensor(features, dtype=torch.float32, device=device)
            if inputs.ndim == 2:
                inputs = inputs.reshape(-1, 3, 32, 32)
            elif inputs.ndim == 4 and inputs.shape[-1] == 3:
                inputs = inputs.permute(0, 3, 1, 2)
            if inputs.ndim != 4 or inputs.shape[1] != 3:
                raise ValueError("MobileNet input must be flattened CIFAR or NCHW/NHWC RGB")
            if float(inputs.detach().max()) > 1.5:
                inputs = inputs / 255.0
            inputs = torch_functional.interpolate(
                inputs, size=(224, 224), mode="bilinear", align_corners=False
            )
            if training:
                if rng is None:
                    raise ValueError("MobileNet training transforms require an RNG")
                inputs = torch_functional.pad(inputs, (28, 28, 28, 28))
                transformed: list[torch.Tensor] = []
                for image in inputs:
                    top = int(rng.integers(0, 57))
                    left = int(rng.integers(0, 57))
                    image = image[:, top:top + 224, left:left + 224]
                    if float(rng.random()) < 0.5:
                        image = torch.flip(image, dims=(2,))
                    transformed.append(image)
                inputs = torch.stack(transformed)
            # These are the CIFAR-10 statistics used by the paper's repository.
            mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device)[None, :, None, None]
            std = torch.tensor([0.2023, 0.1994, 0.2010], device=device)[None, :, None, None]
            inputs = (inputs - mean) / std
        elif name in {"resnet", "resnet18"}:
            inputs = torch.as_tensor(features, dtype=torch.float32, device=device)
            if inputs.ndim == 2:
                inputs = inputs.reshape(-1, 3, 32, 32)
            elif inputs.ndim == 4 and inputs.shape[-1] == 3:
                inputs = inputs.permute(0, 3, 1, 2)
            if inputs.ndim != 4 or inputs.shape[1:] != (3, 32, 32):
                raise ValueError(
                    "ResNet-18 input must be flattened CIFAR or 3x32x32 NCHW/NHWC RGB"
                )
            if float(inputs.detach().max()) > 1.5:
                inputs = inputs / 255.0
            if training:
                if rng is None:
                    raise ValueError("ResNet-18 training transforms require an RNG")
                inputs = torch_functional.pad(inputs, (4, 4, 4, 4))
                transformed: list[torch.Tensor] = []
                for image in inputs:
                    top = int(rng.integers(0, 9))
                    left = int(rng.integers(0, 9))
                    image = image[:, top:top + 32, left:left + 32]
                    if float(rng.random()) < 0.5:
                        image = torch.flip(image, dims=(2,))
                    transformed.append(image)
                inputs = torch.stack(transformed)
            # Population statistics of the CIFAR-100 training set.
            mean = torch.tensor([0.5071, 0.4867, 0.4408], device=device)[None, :, None, None]
            std = torch.tensor([0.2675, 0.2565, 0.2761], device=device)[None, :, None, None]
            inputs = (inputs - mean) / std
        else:
            inputs = torch.as_tensor(features, dtype=torch.float32, device=device)
            if inputs.ndim == 1:
                inputs = inputs[:, None]
        return inputs, labels

    def batches(self, batch_size: int) -> list[np.ndarray]:
        batch_size = len(self) if batch_size == -1 else int(batch_size)
        return [
            np.arange(start, min(start + batch_size, len(self)), dtype=np.int64)
            for start in range(0, len(self), batch_size)
        ]

    def epoch_batches(
        self,
        batch_size: int,
        epochs: int,
        *,
        rng: np.random.Generator | None = None,
        shuffle: bool = False,
    ) -> Iterator[np.ndarray]:
        """Yield complete epochs with the legacy repository's final short batch.

        The published TensorFlow/Ray code resets a client's cursor at the end of
        each epoch instead of filling the final mini-batch from the next epoch.
        Keeping this separate from ``sample_indices`` preserves random exact-step sampling
        for the corrected methods while matching the paper baselines.
        """

        batch_size = len(self) if batch_size == -1 else int(batch_size)
        epochs = int(epochs)
        if batch_size <= 0 or epochs <= 0:
            raise ValueError("batch_size and epochs must be positive (or batch size -1)")
        if shuffle and rng is None:
            raise ValueError("shuffled epoch batches require an RNG")

        base_order = np.arange(len(self), dtype=np.int64)
        for _ in range(epochs):
            order = rng.permutation(len(self)) if shuffle else base_order
            for start in range(0, len(self), batch_size):
                yield np.asarray(order[start : start + batch_size], dtype=np.int64)


@dataclass
class FederatedData:
    clients: dict[int, ClientDataset]
    train_ids: np.ndarray
    validation_ids: np.ndarray
    test_ids: np.ndarray

    @classmethod
    def load(
        cls,
        directory: str | Path,
        fold: int,
        model_name: str,
        inner_fold: int | None = None,
        normalization: str = "none",
        normalized_features: int | None = None,
        sequence_length: int = 80,
        split_prefix: str = "fold",
    ) -> "FederatedData":
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
            candidate_train_ids = np.unique(
                np.concatenate([train_ids, validation_ids])
            )
            inner_validation_ids = _load_ids(
                root / f"fold{int(inner_fold)}_te_client_ids.lst"
            )
            if not np.all(np.isin(inner_validation_ids, candidate_train_ids)):
                raise ValueError("inner validation fold overlaps the outer test fold")
            train_ids = np.setdiff1d(candidate_train_ids, inner_validation_ids)
            validation_ids = inner_validation_ids
        all_ids = np.unique(np.concatenate([train_ids, validation_ids, test_ids]))
        clients = {
            int(client_id): ClientDataset(
                _load_array(root / f"measures_{int(client_id)}"),
                _load_array(root / f"labels_{int(client_id)}"),
                model_name=model_name,
                sequence_length=sequence_length,
            )
            for client_id in all_ids
        }
        result = cls(clients, train_ids, validation_ids, test_ids)
        result.normalize(normalization, normalized_features)
        return result

    def normalize(self, mode: str, normalized_features: int | None = None) -> None:
        mode = mode.lower()
        if mode == "none":
            return
        if mode not in {"local", "global"}:
            raise ValueError("normalization must be none, local, or global")
        if any(
            client.model_name.lower() in {
                *CHARACTER_MODEL_NAMES,
                "mobilenet",
                "mobilenetv2",
                "mbnt",
                "resnet",
                "resnet18",
            }
            for client in self.clients.values()
        ):
            raise ValueError("explicit normalization is only supported for tabular models")

        feature_count = next(iter(self.clients.values())).features.shape[1]
        selected = np.arange(feature_count)
        if normalized_features is not None:
            selected = selected[: int(normalized_features)]

        if mode == "global":
            train_features = np.concatenate(
                [self.clients[int(client_id)].features for client_id in self.train_ids], axis=0
            )
            mean = train_features[:, selected].mean(axis=0)
            std = train_features[:, selected].std(axis=0)
            std[std == 0] = 1.0
            for client in self.clients.values():
                values = client.features.astype(np.float32, copy=True)
                values[:, selected] = (values[:, selected] - mean) / std
                client.features = values
        else:
            for client in self.clients.values():
                values = client.features.astype(np.float32, copy=True)
                mean = values[:, selected].mean(axis=0)
                std = values[:, selected].std(axis=0)
                std[std == 0] = 1.0
                values[:, selected] = (values[:, selected] - mean) / std
                client.features = values

    @property
    def input_size(self) -> int:
        example = self.clients[int(self.train_ids[0])].features
        return int(example.shape[1]) if example.ndim > 1 else 1
