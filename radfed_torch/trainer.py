from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
import random
import time
from typing import Callable, Iterable

import numpy as np
import torch
from torch import nn

from .data import ClientDataset, FederatedData
from .metrics import accuracy_score, binary_auc_score, weighted_f1_score
from .models import average_state_dicts, build_model, clone_state_dict
from .schedule import build_independent_schedule, build_without_replacement_schedule
from .theory import (
    CriticalSchedule,
    ProfileStatistics,
    critical_learning_rate,
    fixed_r_critical_schedule_candidates,
)


def retain_nonincreasing_learning_rate(previous: float, candidate: float) -> float:
    """Accept a smaller candidate rate and retain the previous rate on an increase."""

    previous = float(previous)
    candidate = float(candidate)
    if (
        not math.isfinite(previous)
        or not math.isfinite(candidate)
        or previous <= 0
        or candidate <= 0
    ):
        raise ValueError("learning rates must be positive and finite")
    return min(previous, candidate)


@dataclass
class TrainingConfig:
    method: str = "corrected-fixed"
    split_prefix: str = "fold"
    model_name: str = "ffn"
    num_classes: int = 2
    hidden_size: int = 64
    sequence_length: int = 80
    transformer_layers: int = 4
    transformer_heads: int = 4
    transformer_ff_size: int = 512
    transformer_dropout: float = 0.0
    ffn_hidden_sizes: tuple[int, ...] | list[int] | None = None
    ffn_initialization: str = "kaiming"
    rounds: int = 100
    participation: float = 0.1
    r_local_steps: int = 5
    s_client_visits: int = 5
    local_epochs: int | None = None
    batch_size: int = 256
    eval_batch_size: int = 1024
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    optimizer: str = "sgd"
    metric: str = "loss"
    eval_frequency: int = 1
    training_eval_frequency: int = 0
    profile_frequency: int = 5
    profile_batches: int = 1
    profile_clients: int = 0
    target_epsilon: float | None = None
    l_smooth: float = 1.0
    l_smooth_update: str = "ema"
    t_comm: float = 0.0
    t_comp: float = 0.01
    t_outer: float = 0.0
    fixed_timing_costs: bool = False
    max_total_steps: int = 250
    max_local_steps: int = 25
    seed: int = 1
    device: str = "auto"
    output_dir: str = "out/pytorch_run"
    pretrained_mobilenet: bool = False
    mobilenet_weights_path: str | None = None
    shuffle_client_data: bool = False
    evaluate_test: bool = True
    save_model: bool = True
    evaluate_validation: bool = True
    profile_diagnostics: bool = False

    def validate(self, num_train_clients: int) -> None:
        if num_train_clients <= 0:
            raise ValueError("at least one training client is required")
        allowed = {
            "corrected-fixed",
            "critical",
            "critical-r1",
            "critical-frozen",
            "original-radfed",
            "fedavg",
        }
        if self.method not in allowed:
            raise ValueError(f"method must be one of {sorted(allowed)}")
        if (
            not self.split_prefix
            or Path(self.split_prefix).name != self.split_prefix
            or any(separator in self.split_prefix for separator in ("/", "\\"))
        ):
            raise ValueError("split_prefix must be a nonempty filename prefix")
        if self.num_classes < 2 or self.hidden_size <= 0:
            raise ValueError("num_classes must be at least two and hidden_size positive")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if (
            self.transformer_layers <= 0
            or self.transformer_heads <= 0
            or self.transformer_ff_size <= 0
        ):
            raise ValueError("Transformer layer, head, and feed-forward sizes must be positive")
        if not math.isfinite(self.transformer_dropout) or not 0 <= self.transformer_dropout < 1:
            raise ValueError("transformer_dropout must be in [0, 1)")
        if self.model_name.lower() in {"transformer", "character-transformer"}:
            if self.hidden_size % self.transformer_heads:
                raise ValueError("Transformer hidden size must be divisible by its head count")
            profiled_objective = (
                self.method.startswith("critical")
                or (
                    self.method == "corrected-fixed"
                    and self.l_smooth_update == "raw"
                )
                or self.profile_diagnostics
            )
            if self.transformer_dropout != 0.0 and profiled_objective:
                raise ValueError(
                    "profiled Transformer experiments require zero dropout"
                )
        if self.ffn_hidden_sizes is not None and (
            not self.ffn_hidden_sizes or any(int(value) <= 0 for value in self.ffn_hidden_sizes)
        ):
            raise ValueError("ffn_hidden_sizes must contain positive values")
        if self.ffn_initialization not in {"legacy", "kaiming"}:
            raise ValueError("ffn_initialization must be legacy or kaiming")
        if (
            self.rounds <= 0
            or self.batch_size < -1
            or self.batch_size == 0
            or self.eval_batch_size < -1
            or self.eval_batch_size == 0
        ):
            raise ValueError("rounds and batch sizes must be positive (or batch size -1)")
        if not 0 < self.participation <= 1:
            raise ValueError("participation must be in (0, 1]")
        n_active = max(1, int(self.participation * num_train_clients))
        if self.r_local_steps <= 0:
            raise ValueError("r_local_steps must be positive")
        if (
            self.method in {"corrected-fixed", "critical-frozen", "original-radfed"}
            and not 1 <= self.s_client_visits <= (
                n_active if self.method in {"corrected-fixed", "critical-frozen"} else num_train_clients
            )
        ):
            raise ValueError("s_client_visits exceeds the method's client limit")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if self.optimizer != "sgd":
            raise ValueError("the maintained RADFed implementation requires standard SGD")
        if self.metric.lower() not in {"accuracy", "acc", "f1", "weighted-f1", "auc", "loss"}:
            raise ValueError("metric must be accuracy, f1, auc, or loss")
        if not math.isfinite(self.l_smooth) or self.l_smooth <= 0:
            raise ValueError("l_smooth must be a positive finite initial estimate")
        if self.l_smooth_update not in {"fixed", "raw", "ema"}:
            raise ValueError("l_smooth_update must be fixed, raw, or ema")
        if self.method == "critical-frozen" and self.l_smooth_update != "fixed":
            raise ValueError("critical-frozen requires fixed L")
        if self.profile_diagnostics and (
            self.method not in {"corrected-fixed", "critical-frozen"}
            or self.l_smooth_update != "fixed"
        ):
            raise ValueError("diagnostic-only profiles require a fixed-budget method and fixed L")
        if not self.evaluate_validation and self.evaluate_test:
            raise ValueError("training-only calibration must disable test evaluation too")
        if any(
            not math.isfinite(value) or value < 0
            for value in (self.t_comm, self.t_comp, self.t_outer)
        ):
            raise ValueError("timing inputs must be finite and nonnegative")
        if self.max_total_steps <= 0 or self.max_local_steps <= 0:
            raise ValueError("critical-schedule search limits must be positive")
        if self.method == "critical" and self.r_local_steps > self.max_local_steps:
            raise ValueError("r_local_steps exceeds max_local_steps")
        if self.local_epochs is not None:
            if self.local_epochs <= 0:
                raise ValueError("local_epochs must be positive when supplied")
            if self.method not in {"original-radfed", "fedavg"}:
                raise ValueError("local_epochs is only for original-radfed and fedavg")
        if self.method.startswith("critical") and self.target_epsilon is None:
            raise ValueError("critical methods require an explicit --target-epsilon")
        if self.method.startswith("critical") and n_active <= 1:
            raise ValueError("critical methods require at least two active trajectories")
        if self.profile_batches <= 0:
            raise ValueError("profile_batches must be positive")
        if self.profile_clients < 0:
            raise ValueError("profile_clients must be nonnegative")
        if (
            self.eval_frequency <= 0
            or self.profile_frequency <= 0
            or self.training_eval_frequency < 0
        ):
            raise ValueError(
                "validation/profile frequencies must be positive and "
                "training_eval_frequency must be nonnegative"
            )


class RADFedTrainer:
    def __init__(self, data: FederatedData, config: TrainingConfig):
        self.data = data
        self.config = config
        self.config.validate(len(data.train_ids))
        self.rng = np.random.default_rng(config.seed)
        self.profile_rng = np.random.default_rng(config.seed + 2_000_003)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        self.device = self._resolve_device(config.device)
        self.model_factory: Callable[[], nn.Module] = lambda: build_model(
            model_name=config.model_name,
            input_size=data.input_size,
            hidden_size=config.hidden_size,
            num_classes=config.num_classes,
            ffn_hidden_sizes=config.ffn_hidden_sizes,
            ffn_initialization=config.ffn_initialization,
            vocabulary_size=config.num_classes,
            sequence_length=config.sequence_length,
            transformer_layers=config.transformer_layers,
            transformer_heads=config.transformer_heads,
            transformer_ff_size=config.transformer_ff_size,
            transformer_dropout=config.transformer_dropout,
            pretrained_mobilenet=config.pretrained_mobilenet,
            mobilenet_weights_path=config.mobilenet_weights_path,
        )
        self.model = self.model_factory().to(self.device)
        self.loss_function = nn.CrossEntropyLoss()
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        legacy_best_model = self.output_dir / "best_model.pt"
        if legacy_best_model.is_file():
            legacy_best_model.unlink()
        self.metrics_rows: list[dict[str, object]] = []
        self.profile_rows: list[dict[str, object]] = []
        self.schedule_candidate_rows: list[dict[str, object]] = []
        self.t_comp_ema = float(config.t_comp)
        self.t_outer_ema = float(config.t_outer)
        self.l_smooth_ema = (
            max(1.0, float(config.l_smooth))
            if config.l_smooth_update == "ema"
            else float(config.l_smooth)
        )
        self.previous_profile_gradient: torch.Tensor | None = None
        self.previous_profile_weights: torch.Tensor | None = None
        self.previous_profile_client_gradients: torch.Tensor | None = None
        self.current_learning_rate = float(config.learning_rate)
        profile_count = config.profile_clients or self.num_trajectories
        profile_count = min(profile_count, len(self.data.train_ids))
        profile_cohort_rng = np.random.default_rng(config.seed + 1_000_003)
        self.profile_client_ids = profile_cohort_rng.choice(
            self.data.train_ids, size=profile_count, replace=False
        )
        self.active_client_ids = np.empty(0, dtype=np.int64)
        self.model_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.model.state_dict().values()
        )

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        return device

    @property
    def num_trajectories(self) -> int:
        return max(1, int(self.config.participation * len(self.data.train_ids)))

    def _optimizer(
        self, model: nn.Module, learning_rate: float
    ) -> torch.optim.Optimizer:
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if self.config.optimizer == "sgd":
            return torch.optim.SGD(
                parameters,
                lr=learning_rate,
                weight_decay=0.0,
            )
        raise ValueError("the maintained RADFed implementation requires standard SGD")

    def _optimization_loss(
        self,
        model: nn.Module,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Training objective, including the L2 term used by the optimizer."""

        loss = self.loss_function(logits, labels)
        if self.config.weight_decay:
            penalty = sum(
                torch.sum(parameter * parameter)
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            loss = loss + 0.5 * self.config.weight_decay * penalty
        return loss

    def _train_client_steps(
        self,
        model: nn.Module,
        client: ClientDataset,
        steps: int,
        learning_rate: float,
    ) -> float:
        model.train()
        optimizer = self._optimizer(model, learning_rate)
        start = time.perf_counter()
        for _ in range(steps):
            # Assumption 3 draws fresh stochastic data at every SGD step.  This
            # deliberately does not advance a deterministic client cursor.
            indices = client.sample_indices(self.config.batch_size, self.rng)
            inputs, labels = client.tensors(
                indices, self.device, training=True, rng=self.rng
            )
            optimizer.zero_grad(set_to_none=True)
            loss = self._optimization_loss(model, model(inputs), labels)
            loss.backward()
            optimizer.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter() - start

    def _train_client_epochs(
        self,
        model: nn.Module,
        client: ClientDataset,
        epochs: int,
        learning_rate: float,
    ) -> tuple[float, int]:
        """Run legacy paper epochs, including a short final mini-batch."""

        model.train()
        optimizer = self._optimizer(model, learning_rate)
        start = time.perf_counter()
        steps = 0
        for indices in client.epoch_batches(
            self.config.batch_size,
            epochs,
            rng=self.rng,
            shuffle=self.config.shuffle_client_data,
        ):
            inputs, labels = client.tensors(
                indices, self.device, training=True, rng=self.rng
            )
            optimizer.zero_grad(set_to_none=True)
            loss = self._optimization_loss(model, model(inputs), labels)
            loss.backward()
            optimizer.step()
            steps += 1
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter() - start, steps

    def _steps_for_client(self, client: ClientDataset, corrected_steps: int) -> int:
        if (
            self.config.local_epochs is not None
            and self.config.method in {"original-radfed", "fedavg"}
        ):
            batches_per_epoch = (
                1
                if self.config.batch_size == -1
                else math.ceil(len(client) / self.config.batch_size)
            )
            return self.config.local_epochs * batches_per_epoch
        return corrected_steps

    def _schedule(self, s_visits: int) -> np.ndarray:
        if self.config.method in {"corrected-fixed", "critical", "critical-r1", "critical-frozen"}:
            # The finite population in Phi(S) is the active N-client cohort.
            # Sample that cohort once and keep it fixed for every redistribution
            # in this outer round.  A new cohort is drawn next outer round so all
            # training clients remain eligible over the full experiment.
            self.active_client_ids = self.rng.choice(
                self.data.train_ids,
                size=self.num_trajectories,
                replace=False,
            ).astype(np.int64)
            return build_without_replacement_schedule(
                self.active_client_ids,
                self.num_trajectories,
                s_visits,
                self.rng,
            ).assignments
        if self.config.method == "fedavg":
            return build_independent_schedule(
                self.data.train_ids,
                self.num_trajectories,
                1,
                self.rng,
            )
        return build_independent_schedule(
            self.data.train_ids,
            self.num_trajectories,
            s_visits,
            self.rng,
        )

    def _run_outer_round(
        self,
        global_state: OrderedDict[str, torch.Tensor],
        r_steps: int,
        s_visits: int,
        learning_rate: float,
        visit_callback: Callable[
            [
                int,
                OrderedDict[str, torch.Tensor],
                float,
                float,
                int,
                int,
            ],
            None,
        ]
        | None = None,
    ) -> tuple[OrderedDict[str, torch.Tensor], float, float, int, int]:
        assignments = self._schedule(s_visits)
        trajectories = [
            OrderedDict((key, value.clone()) for key, value in global_state.items())
            for _ in range(self.num_trajectories)
        ]
        worker = self.model_factory().to(self.device)
        redistribution_seconds = 0.0
        critical_compute_seconds = 0.0
        total_client_steps = 0
        trajectory_steps = [0 for _ in range(self.num_trajectories)]

        for visit in range(assignments.shape[0]):
            visit_times: list[float] = []
            visit_client_steps = 0
            visit_trajectory_steps: list[int] = []
            for trajectory_index, client_id in enumerate(assignments[visit]):
                worker.load_state_dict(trajectories[trajectory_index], strict=True)
                client = self.data.clients[int(client_id)]
                if (
                    self.config.local_epochs is not None
                    and self.config.method in {"original-radfed", "fedavg"}
                ):
                    elapsed, client_steps = self._train_client_epochs(
                        worker,
                        client,
                        self.config.local_epochs,
                        learning_rate,
                    )
                else:
                    client_steps = self._steps_for_client(client, r_steps)
                    elapsed = self._train_client_steps(
                        worker,
                        client,
                        client_steps,
                        learning_rate,
                    )
                visit_times.append(elapsed)
                visit_client_steps += client_steps
                visit_trajectory_steps.append(client_steps)
                total_client_steps += client_steps
                trajectory_steps[trajectory_index] += client_steps
                trajectories[trajectory_index] = clone_state_dict(worker)
            critical_compute_seconds += max(visit_times)
            redistribution_seconds += max(visit_times) + self.config.t_comm

            if visit_callback is not None:
                visit_weights = None
                if self.config.method == "fedavg":
                    visit_weights = [
                        len(self.data.clients[int(client_id)])
                        for client_id in assignments[visit]
                    ]
                visit_callback(
                    visit + 1,
                    average_state_dicts(trajectories, visit_weights),
                    sum(visit_times),
                    max(visit_times),
                    visit_client_steps,
                    max(visit_trajectory_steps),
                )

        aggregation_weights = None
        if self.config.method == "fedavg":
            aggregation_weights = [
                len(self.data.clients[int(client_id)]) for client_id in assignments[0]
            ]
        return (
            average_state_dicts(trajectories, aggregation_weights),
            redistribution_seconds,
            critical_compute_seconds,
            total_client_steps,
            max(trajectory_steps),
        )

    def _predict_client(
        self,
        client: ClientDataset,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        self.model.eval()
        labels_all: list[np.ndarray] = []
        predictions_all: list[np.ndarray] = []
        positive_scores: list[np.ndarray] = []
        weighted_loss = 0.0
        count = 0
        with torch.no_grad():
            for indices in client.batches(self.config.eval_batch_size):
                inputs, labels = client.tensors(indices, self.device)
                logits = self.model(inputs)
                probabilities = torch.softmax(logits, dim=1)
                loss = self.loss_function(logits, labels)
                batch_count = len(indices)
                weighted_loss += float(loss) * batch_count
                count += batch_count
                labels_all.append(labels.detach().cpu().numpy())
                predictions_all.append(torch.argmax(logits, dim=1).cpu().numpy())
                if probabilities.shape[1] >= 2:
                    positive_scores.append(probabilities[:, 1].cpu().numpy())
        labels_np = np.concatenate(labels_all)
        predictions_np = np.concatenate(predictions_all)
        scores_np = np.concatenate(positive_scores) if positive_scores else np.empty(0)
        return labels_np, predictions_np, scores_np, weighted_loss / count

    def evaluate(self, client_ids: Iterable[int]) -> tuple[float, float]:
        scores: list[float] = []
        losses: list[float] = []
        metric = self.config.metric.lower()
        for client_id in client_ids:
            labels, predictions, positive_scores, loss = self._predict_client(
                self.data.clients[int(client_id)]
            )
            if metric in {"accuracy", "acc", "loss"}:
                score = accuracy_score(labels, predictions)
            elif metric in {"f1", "weighted-f1"}:
                score = weighted_f1_score(labels, predictions)
            elif metric == "auc":
                try:
                    score = binary_auc_score(labels, positive_scores)
                except ValueError:
                    continue
            else:
                raise ValueError("metric must be accuracy, f1, auc, or loss")
            scores.append(score)
            losses.append(loss)
        if not scores:
            raise ValueError("no evaluable clients for the requested metric")
        return float(np.mean(scores)), float(np.mean(losses))

    @staticmethod
    def _flatten_gradients(
        gradients: tuple[torch.Tensor | None, ...],
        parameters: list[torch.Tensor],
    ) -> torch.Tensor:
        values = [
            (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
            for gradient, parameter in zip(gradients, parameters)
        ]
        return torch.cat(values)

    @staticmethod
    def _flatten_parameters(model: nn.Module) -> torch.Tensor:
        return torch.cat(
            [
                parameter.detach().cpu().to(torch.float64).reshape(-1)
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
        )

    def _batch_gradient(
        self,
        model: nn.Module,
        client: ClientDataset,
        indices: np.ndarray,
        training: bool = False,
        rng: np.random.Generator | None = None,
    ) -> torch.Tensor:
        transform_rng = (self.rng if rng is None else rng) if training else None
        inputs, labels = client.tensors(
            indices,
            self.device,
            training=training,
            rng=transform_rng,
        )
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        loss = self._optimization_loss(model, model(inputs), labels)
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        return self._flatten_gradients(gradients, parameters).detach().cpu()

    def _full_gradient(self, model: nn.Module, client: ClientDataset) -> torch.Tensor:
        total = len(client)
        accumulator: torch.Tensor | None = None
        batch_size = self.config.eval_batch_size
        if self.config.batch_size > 0 and any(
            isinstance(module, (nn.RNNBase, nn.TransformerEncoder))
            for module in model.modules()
        ):
            # Sequence-model backward retains substantially more state than
            # evaluation. Smaller chunks compute the same sample-weighted full
            # gradient without risking a large profiling allocation.
            batch_size = min(batch_size, self.config.batch_size)
        for indices in client.batches(batch_size):
            batch_gradient = self._batch_gradient(model, client, indices)
            if accumulator is None:
                accumulator = torch.zeros_like(batch_gradient, dtype=torch.float64)
            accumulator += batch_gradient.to(torch.float64) * (len(indices) / total)
        if accumulator is None:
            raise ValueError("cannot profile an empty client")
        return accumulator

    @staticmethod
    def _prepare_gradient_model(model: nn.Module) -> None:
        """Use deterministic evaluation semantics while retaining RNN backward."""

        model.eval()
        for module in model.modules():
            if isinstance(module, nn.RNNBase):
                # cuDNN requires its training forward path for RNN backward.
                # The profiling/calibration model is disposable, and recurrent
                # dropout is disabled so both secant gradients use one objective.
                module.train()
                module.dropout = 0.0

    def _profile_global_gradient(self, model: nn.Module) -> torch.Tensor:
        gradients = [
            self._full_gradient(model, self.data.clients[int(client_id)])
            for client_id in self.profile_client_ids
        ]
        if not gradients:
            raise ValueError("cannot estimate L without profiling clients")
        return torch.mean(torch.stack(gradients), dim=0)

    def estimate_l_smooth_secant(
        self, relative_probe_distance: float
    ) -> dict[str, object]:
        """Estimate local smoothness from a discarded training-only secant probe."""

        if (
            not math.isfinite(relative_probe_distance)
            or relative_probe_distance <= 0
        ):
            raise ValueError("relative_probe_distance must be positive and finite")
        model = self.model_factory().to(self.device)
        model.load_state_dict(self.model.state_dict())
        self._prepare_gradient_model(model)
        weights_before = self._flatten_parameters(model)
        gradient_before = self._profile_global_gradient(model)
        gradient_norm = torch.linalg.vector_norm(gradient_before)
        if not math.isfinite(float(gradient_norm)) or float(gradient_norm) <= 0:
            raise ValueError("the initial profiling gradient has zero or invalid norm")
        weight_norm = torch.linalg.vector_norm(weights_before)
        probe_distance = relative_probe_distance * max(float(weight_norm), 1.0)
        direction = -gradient_before / gradient_norm
        offset = 0
        with torch.no_grad():
            for parameter in model.parameters():
                if not parameter.requires_grad:
                    continue
                count = parameter.numel()
                step = direction[offset : offset + count].reshape(parameter.shape)
                parameter.add_(
                    step.to(device=parameter.device, dtype=parameter.dtype),
                    alpha=probe_distance,
                )
                offset += count
        if offset != direction.numel():
            raise RuntimeError("secant probe did not consume every parameter")
        weights_after = self._flatten_parameters(model)
        gradient_after = self._profile_global_gradient(model)
        weight_delta = torch.linalg.vector_norm(weights_after - weights_before)
        gradient_delta = torch.linalg.vector_norm(gradient_after - gradient_before)
        denominator = float(weight_delta)
        numerator = float(gradient_delta)
        if denominator <= 0 or not math.isfinite(denominator):
            raise ValueError("secant probe produced an invalid parameter displacement")
        l_raw = numerator / denominator
        if not math.isfinite(l_raw) or l_raw <= 0:
            raise ValueError("secant probe produced a nonpositive or invalid L estimate")
        return {
            "schema_version": 1,
            "estimator": "gradient_secant",
            "formula": "||grad F(w_probe)-grad F(w_0)||/||w_probe-w_0||",
            "objective": "client-balanced mean over a fixed training-client cohort",
            "test_clients_used": False,
            "validation_clients_used": False,
            "profile_client_ids": [int(value) for value in self.profile_client_ids],
            "profile_clients": int(len(self.profile_client_ids)),
            "relative_probe_distance": float(relative_probe_distance),
            "absolute_probe_distance": denominator,
            "initial_parameter_norm": float(weight_norm),
            "initial_gradient_norm": float(gradient_norm),
            "gradient_difference_norm": numerator,
            "parameter_difference_norm": denominator,
            "l_raw": l_raw,
            "model_reset_after_probe": True,
            "batch_norm_semantics": "evaluation mode with fixed running statistics",
        }

    def profile(self, round_number: int) -> ProfileStatistics:
        selected_ids = self.profile_client_ids
        model = self.model_factory().to(self.device)
        model.load_state_dict(self.model.state_dict())
        self._prepare_gradient_model(model)
        current_weights = self._flatten_parameters(model)
        full_gradients: list[torch.Tensor] = []
        sigma_samples: list[float] = []
        start = time.perf_counter()
        for client_id in selected_ids:
            client = self.data.clients[int(client_id)]
            full_gradient = self._full_gradient(model, client)
            full_gradients.append(full_gradient)
            for _ in range(self.config.profile_batches):
                indices = client.sample_indices(
                    self.config.batch_size, self.profile_rng
                )
                stochastic = self._batch_gradient(
                    model,
                    client,
                    indices,
                    training=True,
                    rng=self.profile_rng,
                ).to(torch.float64)
                difference = stochastic - full_gradient
                sigma_samples.append(float(torch.dot(difference, difference)))
        stacked = torch.stack(full_gradients)
        global_gradient = torch.mean(stacked, dim=0)
        g_sq = float(torch.mean(torch.sum(stacked * stacked, dim=1)))
        sigma_sq = float(np.mean(sigma_samples))
        l_raw: float | None = None
        l_raw_max_client: float | None = None
        if (
            self.previous_profile_gradient is not None
            and self.previous_profile_weights is not None
        ):
            weight_delta = torch.linalg.vector_norm(
                current_weights - self.previous_profile_weights
            )
            if float(weight_delta) > 0:
                gradient_delta = torch.linalg.vector_norm(
                    global_gradient - self.previous_profile_gradient
                )
                l_raw = float(gradient_delta / weight_delta)
                if self.previous_profile_client_gradients is not None:
                    l_raw_max_client = float(torch.max(torch.linalg.vector_norm(
                        stacked - self.previous_profile_client_gradients, dim=1
                    )) / weight_delta)
                if self.config.l_smooth_update == "raw":
                    if not math.isfinite(l_raw) or l_raw <= 0:
                        raise ValueError("profile produced a nonpositive or invalid L estimate")
                    self.l_smooth_ema = l_raw
                elif self.config.l_smooth_update == "ema":
                    self.l_smooth_ema = max(
                        1.0, 0.8 * self.l_smooth_ema + 0.2 * l_raw
                    )
        self.previous_profile_gradient = global_gradient.clone()
        self.previous_profile_weights = current_weights.clone()
        if self.config.profile_diagnostics:
            self.previous_profile_client_gradients = stacked.clone()
        elapsed = time.perf_counter() - start
        profile = ProfileStatistics(
            sigma_sq=sigma_sq,
            g_sq=g_sq,
            l_smooth=self.l_smooth_ema,
        )
        self.profile_rows.append(
            {
                "round": round_number,
                "profile_clients": len(selected_ids),
                "profile_batches": self.config.profile_batches,
                "sigma_sq": sigma_sq,
                "g_sq": g_sq,
                "l_raw": l_raw,
                "l_raw_max_client": l_raw_max_client,
                "l_smooth": self.l_smooth_ema,
                "profile_seconds": elapsed,
            }
        )
        return profile

    def _select_critical_schedule(
        self,
        profile: ProfileStatistics,
        profile_round: int,
    ) -> CriticalSchedule:
        fixed_r_steps = (
            1
            if self.config.method == "critical-r1"
            else self.config.r_local_steps
        )
        candidates = fixed_r_critical_schedule_candidates(
            profile=profile,
            n_active=self.num_trajectories,
            target_epsilon=float(self.config.target_epsilon),
            r_local_steps=fixed_r_steps,
            t_comm=self.config.t_comm,
            t_comp=self.t_comp_ema,
            max_total_steps=self.config.max_total_steps,
        )
        for candidate in candidates:
            self.schedule_candidate_rows.append(
                {
                    "profile_round": profile_round,
                    "l_smooth": profile.l_smooth,
                    **asdict(candidate),
                }
            )
        return min(
            candidates,
            key=lambda item: (
                item.normalized_cost,
                item.round_seconds,
                item.e_total_steps,
                item.s_client_visits,
                item.r_local_steps,
            ),
        )

    def _write_outputs(self) -> None:
        with (self.output_dir / "round_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            if self.metrics_rows:
                writer = csv.DictWriter(stream, fieldnames=list(self.metrics_rows[0]))
                writer.writeheader()
                writer.writerows(self.metrics_rows)
        with (self.output_dir / "profile_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            if self.profile_rows:
                writer = csv.DictWriter(stream, fieldnames=list(self.profile_rows[0]))
                writer.writeheader()
                writer.writerows(self.profile_rows)
        with (self.output_dir / "schedule_candidates.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            if self.schedule_candidate_rows:
                writer = csv.DictWriter(
                    stream, fieldnames=list(self.schedule_candidate_rows[0])
                )
                writer.writeheader()
                writer.writerows(self.schedule_candidate_rows)
        with (self.output_dir / "config.json").open("w", encoding="utf-8") as stream:
            json.dump(asdict(self.config), stream, indent=2, sort_keys=True)
        checkpoint = self.output_dir / "current_model.pt"
        if self.config.save_model:
            torch.save(clone_state_dict(self.model), checkpoint)
        else:
            checkpoint.unlink(missing_ok=True)

    def train(self) -> dict[str, float | int | str | bool | None]:
        global_state = clone_state_dict(self.model)
        # Include the initial critical-profile overhead in paper-facing wall
        # time so comparisons with non-profiled baselines are complete.
        training_start = time.perf_counter()
        r_steps = self.config.r_local_steps
        s_visits = 1 if self.config.method == "fedavg" else self.config.s_client_visits
        if self.config.method in {"corrected-fixed", "critical-frozen"}:
            self.current_learning_rate = critical_learning_rate(
                self.l_smooth_ema, r_steps * s_visits
            )
        else:
            self.current_learning_rate = self.config.learning_rate
        latest_profile: ProfileStatistics | None = None
        if self.config.method in {"critical", "critical-r1"}:
            latest_profile = self.profile(round_number=0)
            try:
                selected = self._select_critical_schedule(
                    latest_profile, profile_round=0
                )
            except ValueError:
                self._write_outputs()
                raise
            r_steps = selected.r_local_steps
            s_visits = selected.s_client_visits
            self.current_learning_rate = selected.learning_rate
        elif self.config.profile_diagnostics or (
            self.config.method == "corrected-fixed"
            and self.config.l_smooth_update == "raw"
        ):
            # Establish the previous (w, grad F(w)) pair. The first raw
            # secant update becomes available at the next profiling round.
            latest_profile = self.profile(round_number=0)

        cumulative_redistribution_stages = 0
        cumulative_client_steps = 0
        cumulative_transmitted_bytes = 0
        cumulative_training_evaluation_seconds = 0.0
        paper_baseline = self.config.method in {"original-radfed", "fedavg"}
        local_stage_index = 0
        for round_index in range(1, self.config.rounds + 1):
            round_r_steps = r_steps
            round_s_visits = s_visits
            round_learning_rate = self.current_learning_rate
            outer_start = time.perf_counter()
            simulator_train_start = time.perf_counter()

            visit_callback = None
            if paper_baseline:

                def record_paper_stage(
                    visit_index: int,
                    visit_state: OrderedDict[str, torch.Tensor],
                    simulator_visit_seconds: float,
                    critical_visit_seconds: float,
                    visit_client_steps: int,
                    max_client_steps: int,
                ) -> None:
                    nonlocal local_stage_index
                    nonlocal cumulative_redistribution_stages
                    nonlocal cumulative_client_steps
                    nonlocal cumulative_transmitted_bytes
                    nonlocal cumulative_training_evaluation_seconds

                    local_stage_index += 1
                    cumulative_redistribution_stages += 1
                    cumulative_client_steps += visit_client_steps
                    stage_transmitted_bytes = 2 * self.num_trajectories * self.model_bytes
                    cumulative_transmitted_bytes += stage_transmitted_bytes
                    observed_t_comp = critical_visit_seconds / max(1, max_client_steps)
                    if not self.config.fixed_timing_costs:
                        self.t_comp_ema = (
                            0.8 * self.t_comp_ema + 0.2 * observed_t_comp
                        )

                    validation_score: float | None = None
                    validation_loss: float | None = None
                    training_score: float | None = None
                    training_loss: float | None = None
                    evaluation_seconds = 0.0
                    training_evaluation_seconds = 0.0
                    should_evaluate = (
                        local_stage_index % self.config.eval_frequency == 0
                        or (
                            round_index == self.config.rounds
                            and visit_index == round_s_visits
                        )
                    )
                    if should_evaluate:
                        self.model.load_state_dict(visit_state)
                        evaluation_start = time.perf_counter()
                        validation_score, validation_loss = self.evaluate(
                            self.data.validation_ids
                        )
                        evaluation_seconds = time.perf_counter() - evaluation_start
                    should_evaluate_training = (
                        self.config.training_eval_frequency > 0
                        and visit_index == round_s_visits
                        and (
                            round_index % self.config.training_eval_frequency == 0
                            or round_index == self.config.rounds
                        )
                    )
                    if should_evaluate_training:
                        self.model.load_state_dict(visit_state)
                        training_evaluation_start = time.perf_counter()
                        training_score, training_loss = self.evaluate(
                            self.data.train_ids
                        )
                        training_evaluation_seconds = (
                            time.perf_counter() - training_evaluation_start
                        )
                        cumulative_training_evaluation_seconds += (
                            training_evaluation_seconds
                        )
                    self.metrics_rows.append(
                        {
                            "round": local_stage_index,
                            "outer_round": round_index,
                            "visit_in_outer": visit_index,
                            "is_aggregation": visit_index == round_s_visits,
                            "method": self.config.method,
                            "validation_score": validation_score,
                            "validation_loss": validation_loss,
                            "training_score": training_score,
                            "training_loss": training_loss,
                            "r_local_steps": round_r_steps,
                            "s_client_visits": round_s_visits,
                            "e_total_steps": None,
                            "local_epochs": self.config.local_epochs,
                            "learning_rate": round_learning_rate,
                            "active_trajectories": self.num_trajectories,
                            "round_client_sgd_steps": visit_client_steps,
                            "max_trajectory_sgd_steps": max_client_steps,
                            "cumulative_client_sgd_steps": cumulative_client_steps,
                            "redistribution_stages": 1,
                            "cumulative_redistribution_stages": cumulative_redistribution_stages,
                            "client_model_transfers": self.num_trajectories,
                            "aggregation_count": (
                                round_index
                                if visit_index == round_s_visits
                                else round_index - 1
                            ),
                            "round_transmitted_bytes": stage_transmitted_bytes,
                            "cumulative_transmitted_bytes": cumulative_transmitted_bytes,
                            "wall_round_seconds": simulator_visit_seconds
                            + evaluation_seconds,
                            "wall_total_seconds": (
                                time.perf_counter()
                                - training_start
                                - cumulative_training_evaluation_seconds
                            ),
                            "wall_time_includes_initial_profile": True,
                            "wall_time_excludes_training_evaluation": True,
                            "simulator_train_seconds": simulator_visit_seconds,
                            "modeled_critical_round_seconds": (
                                critical_visit_seconds + self.config.t_comm
                            ),
                            "critical_compute_seconds": critical_visit_seconds,
                            "evaluation_seconds": evaluation_seconds,
                            "training_evaluation_seconds": training_evaluation_seconds,
                            "cumulative_training_evaluation_seconds": (
                                cumulative_training_evaluation_seconds
                            ),
                            "profile_seconds": 0.0,
                            "t_comm_seconds": self.config.t_comm,
                            "t_comp_ema_seconds": self.t_comp_ema,
                            "t_outer_ema_seconds": self.t_outer_ema,
                            "sigma_sq": None,
                            "g_sq": None,
                            "l_smooth": self.l_smooth_ema,
                            "target_epsilon": self.config.target_epsilon,
                        }
                    )

                visit_callback = record_paper_stage

            (
                next_state,
                redistribution_seconds,
                critical_compute_seconds,
                round_client_steps,
                max_trajectory_steps,
            ) = self._run_outer_round(
                global_state,
                round_r_steps,
                round_s_visits,
                round_learning_rate,
                visit_callback,
            )
            simulator_train_seconds = time.perf_counter() - simulator_train_start
            global_state = next_state
            self.model.load_state_dict(global_state)
            if paper_baseline:
                self._write_outputs()
                continue
            cumulative_redistribution_stages += round_s_visits
            cumulative_client_steps += round_client_steps
            round_transmitted_bytes = (
                2 * self.num_trajectories * round_s_visits * self.model_bytes
            )
            cumulative_transmitted_bytes += round_transmitted_bytes
            observed_t_comp = critical_compute_seconds / max(
                1, max_trajectory_steps
            )
            if not self.config.fixed_timing_costs:
                self.t_comp_ema = 0.8 * self.t_comp_ema + 0.2 * observed_t_comp

            validation_score: float | None = None
            validation_loss: float | None = None
            training_score: float | None = None
            training_loss: float | None = None
            evaluation_seconds = 0.0
            training_evaluation_seconds = 0.0
            if self.config.evaluate_validation and (
                round_index % self.config.eval_frequency == 0 or round_index == self.config.rounds
            ):
                evaluation_start = time.perf_counter()
                validation_score, validation_loss = self.evaluate(self.data.validation_ids)
                evaluation_seconds = time.perf_counter() - evaluation_start
            if self.config.training_eval_frequency > 0 and (
                round_index % self.config.training_eval_frequency == 0
                or round_index == self.config.rounds
            ):
                training_evaluation_start = time.perf_counter()
                training_score, training_loss = self.evaluate(self.data.train_ids)
                training_evaluation_seconds = (
                    time.perf_counter() - training_evaluation_start
                )
                cumulative_training_evaluation_seconds += training_evaluation_seconds
            profile_seconds = 0.0
            profile_error: ValueError | None = None
            proposed_next_learning_rate = round_learning_rate
            learning_rate_increase_blocked = False
            if (
                (
                    self.config.profile_diagnostics
                    or self.config.method in {"critical", "critical-r1"}
                    or (
                        self.config.method == "corrected-fixed"
                        and self.config.l_smooth_update == "raw"
                    )
                )
                and round_index % self.config.profile_frequency == 0
                and (round_index < self.config.rounds or self.config.profile_diagnostics)
            ):
                profile_start = time.perf_counter()
                latest_profile = self.profile(round_number=round_index)
                if self.config.method in {"critical", "critical-r1"}:
                    try:
                        selected = self._select_critical_schedule(
                            latest_profile, profile_round=round_index
                        )
                    except ValueError as error:
                        profile_error = error
                    else:
                        r_steps = selected.r_local_steps
                        s_visits = selected.s_client_visits
                        proposed_next_learning_rate = selected.learning_rate
                        learning_rate_increase_blocked = (
                            proposed_next_learning_rate > self.current_learning_rate
                        )
                        self.current_learning_rate = retain_nonincreasing_learning_rate(
                            self.current_learning_rate,
                            proposed_next_learning_rate,
                        )
                elif not self.config.profile_diagnostics:
                    proposed_next_learning_rate = critical_learning_rate(
                        latest_profile.l_smooth, r_steps * s_visits
                    )
                    learning_rate_increase_blocked = (
                        proposed_next_learning_rate > self.current_learning_rate
                    )
                    self.current_learning_rate = retain_nonincreasing_learning_rate(
                        self.current_learning_rate,
                        proposed_next_learning_rate,
                    )
                profile_seconds = time.perf_counter() - profile_start

            wall_round = (
                time.perf_counter() - outer_start - training_evaluation_seconds
            )
            self.metrics_rows.append(
                {
                    "round": round_index,
                    "outer_round": round_index,
                    "visit_in_outer": round_s_visits,
                    "is_aggregation": True,
                    "method": self.config.method,
                    "validation_score": validation_score,
                    "validation_loss": validation_loss,
                    "training_score": training_score,
                    "training_loss": training_loss,
                    "r_local_steps": round_r_steps,
                    "s_client_visits": round_s_visits,
                    "e_total_steps": (
                        None
                        if self.config.local_epochs is not None
                        else round_r_steps * round_s_visits
                    ),
                    "local_epochs": self.config.local_epochs,
                    "learning_rate": round_learning_rate,
                    "proposed_next_learning_rate": proposed_next_learning_rate,
                    "next_learning_rate": self.current_learning_rate,
                    "learning_rate_increase_blocked": learning_rate_increase_blocked,
                    "active_trajectories": self.num_trajectories,
                    "round_client_sgd_steps": round_client_steps,
                    "max_trajectory_sgd_steps": max_trajectory_steps,
                    "cumulative_client_sgd_steps": cumulative_client_steps,
                    "redistribution_stages": round_s_visits,
                    "cumulative_redistribution_stages": cumulative_redistribution_stages,
                    "client_model_transfers": self.num_trajectories * round_s_visits,
                    "aggregation_count": round_index,
                    "round_transmitted_bytes": round_transmitted_bytes,
                    "cumulative_transmitted_bytes": cumulative_transmitted_bytes,
                    "wall_round_seconds": wall_round,
                    "wall_total_seconds": (
                        time.perf_counter()
                        - training_start
                        - cumulative_training_evaluation_seconds
                    ),
                    "wall_time_includes_initial_profile": True,
                    "wall_time_excludes_training_evaluation": True,
                    "simulator_train_seconds": simulator_train_seconds,
                    "modeled_critical_round_seconds": (
                        self.t_outer_ema + redistribution_seconds
                    ),
                    "critical_compute_seconds": critical_compute_seconds,
                    "evaluation_seconds": evaluation_seconds,
                    "training_evaluation_seconds": training_evaluation_seconds,
                    "cumulative_training_evaluation_seconds": (
                        cumulative_training_evaluation_seconds
                    ),
                    "profile_seconds": profile_seconds,
                    "t_comm_seconds": self.config.t_comm,
                    "t_comp_ema_seconds": self.t_comp_ema,
                    "t_outer_ema_seconds": self.t_outer_ema,
                    "sigma_sq": None if latest_profile is None else latest_profile.sigma_sq,
                    "g_sq": None if latest_profile is None else latest_profile.g_sq,
                    "l_smooth": self.l_smooth_ema,
                    "target_epsilon": self.config.target_epsilon,
                }
            )
            self._write_outputs()
            if profile_error is not None:
                raise profile_error

        current_metrics = next(
            row
            for row in reversed(self.metrics_rows)
            if bool(row["is_aggregation"])
            and (not self.config.evaluate_validation or (
                row["validation_score"] is not None and row["validation_loss"] is not None
            ))
        )
        aggregation_metrics = [
            row for row in self.metrics_rows if bool(row["is_aggregation"])
        ]
        learning_rates = [float(row["learning_rate"]) for row in aggregation_metrics]
        learning_rate_nonincreasing = all(
            later <= earlier
            for earlier, later in zip(learning_rates, learning_rates[1:])
        )
        if not learning_rate_nonincreasing:
            raise RuntimeError("recorded aggregation learning rates increased")
        if self.config.evaluate_test:
            test_score, test_loss = self.evaluate(self.data.test_ids)
        else:
            test_score, test_loss = None, None
        result = {
            "method": self.config.method,
            "l_smooth_update": self.config.l_smooth_update,
            "initial_l_smooth": float(self.config.l_smooth),
            "final_l_smooth": float(self.l_smooth_ema),
            "evaluation_checkpoint": "current_final",
            "test_evaluated": self.config.evaluate_test,
            "validation_evaluated": self.config.evaluate_validation,
            "training_evaluated": self.config.training_eval_frequency > 0,
            "current_round": int(current_metrics["round"]),
            "current_outer_round": int(current_metrics["outer_round"]),
            "current_visit_in_outer": int(current_metrics["visit_in_outer"]),
            "current_validation_score": (
                float(current_metrics["validation_score"]) if self.config.evaluate_validation else None
            ),
            "current_validation_loss": (
                float(current_metrics["validation_loss"]) if self.config.evaluate_validation else None
            ),
            "current_training_score": (
                float(current_metrics["training_score"])
                if self.config.training_eval_frequency > 0
                else None
            ),
            "current_training_loss": (
                float(current_metrics["training_loss"])
                if self.config.training_eval_frequency > 0
                else None
            ),
            "current_learning_rate": float(current_metrics["learning_rate"]),
            "next_learning_rate": float(
                current_metrics.get("next_learning_rate", current_metrics["learning_rate"])
            ),
            "learning_rate_nonincreasing": learning_rate_nonincreasing,
            "learning_rate_increase_blocks": sum(
                bool(row.get("learning_rate_increase_blocked", False))
                for row in aggregation_metrics
            ),
            "test_score": test_score,
            "test_loss": test_loss,
            "local_training_stages": (
                local_stage_index
                if paper_baseline
                else cumulative_redistribution_stages
            ),
            "aggregation_count": self.config.rounds,
            "client_sgd_steps": cumulative_client_steps,
        }
        with (self.output_dir / "results.json").open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
        self._write_outputs()
        return result
