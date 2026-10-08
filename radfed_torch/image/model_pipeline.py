"""Model pipeline: construct the requested architecture and initialize its weights.

The trainer supplies the data dimensions and seed before calling build_model.
State copying and averaging retain the model's tensor and buffer conventions."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class LogisticRegression(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


class CharacterLSTM(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        num_classes: int,
        hidden_size: int = 256,
        embedding_size: int = 8,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(inputs)
        output, _ = self.lstm(embedded)
        return self.classifier(output[:, -1, :])


def build_model(
    *,
    model_name: str,
    input_size: int,
    hidden_size: int,
    num_classes: int,
    vocabulary_size: int = 80,
    pretrained_mobilenet: bool = False,
    mobilenet_weights_path: str | None = None,
) -> nn.Module:
    name = model_name.strip().lower()
    if name in {"ffn", "mlp"}:
        return FeedForwardNetwork(input_size, hidden_size, num_classes)
    if name in {"lr", "logistic"}:
        return LogisticRegression(input_size, num_classes)
    if name in {"lstm", "shakespeare"}:
        return CharacterLSTM(vocabulary_size, num_classes, hidden_size)
    if name in {"mobilenet", "mobilenetv2", "mbnt"}:
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

        # A supplied state file makes Nomad runs independent of outbound
        # internet access. Only ask torchvision to download weights when no
        # local file was supplied explicitly.
        weights = (
            MobileNet_V2_Weights.DEFAULT
            if pretrained_mobilenet and not mobilenet_weights_path
            else None
        )
        model = mobilenet_v2(weights=weights)
        if mobilenet_weights_path:
            weights_path = Path(mobilenet_weights_path)
            if not weights_path.is_file():
                raise FileNotFoundError(
                    f"MobileNet weights do not exist: {weights_path}"
                )
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        if pretrained_mobilenet or mobilenet_weights_path:
            for parameter in model.parameters():
                parameter.requires_grad = False
            for block in model.features[-2:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            for parameter in model.classifier.parameters():
                parameter.requires_grad = True
        return model
    if name in {"resnet", "resnet18"}:
        from torchvision.models import resnet18

        # The ImageNet stem discards too much spatial information from 32x32
        # CIFAR images.  This is the standard CIFAR adaptation of ResNet-18:
        # retain all four residual stages, but use a 3x3 stride-one stem and
        # remove the initial max-pooling layer.  The model is trained from
        # scratch so the experiment does not depend on downloaded weights.
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"unsupported model_name: {model_name}")


def clone_state_dict(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value.detach().cpu().clone())
        for name, value in model.state_dict().items()
    )


def average_state_dicts(
    states: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float] | None = None,
) -> OrderedDict[str, torch.Tensor]:
    if not states:
        raise ValueError("at least one model state is required")
    keys = tuple(states[0].keys())
    if any(tuple(state.keys()) != keys for state in states[1:]):
        raise ValueError("all model states must have identical keys")

    if weights is None:
        normalized = torch.full((len(states),), 1.0 / len(states), dtype=torch.float64)
    else:
        normalized = torch.as_tensor(weights, dtype=torch.float64)
        if normalized.numel() != len(states) or torch.any(normalized < 0):
            raise ValueError("aggregation weights must be nonnegative and match states")
        total = float(normalized.sum())
        if total <= 0:
            raise ValueError("aggregation weights must have positive sum")
        normalized = normalized / total

    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in keys:
        first = states[0][key]
        if first.is_floating_point() or first.is_complex():
            accumulator = torch.zeros_like(first, device="cpu")
            for coefficient, state in zip(normalized, states):
                accumulator.add_(state[key].detach().cpu(), alpha=float(coefficient))
            result[key] = accumulator
        else:
            result[key] = first.detach().cpu().clone()
    return result
