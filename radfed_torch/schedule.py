from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClientPathSchedule:
    """A cyclic Latin schedule satisfying the consolidated theory."""

    permutation: np.ndarray
    offsets: np.ndarray
    assignments: np.ndarray

    @property
    def num_visits(self) -> int:
        return int(self.assignments.shape[0])

    @property
    def num_trajectories(self) -> int:
        return int(self.assignments.shape[1])

    def validate(self) -> None:
        if self.assignments.ndim != 2:
            raise ValueError("assignments must have shape [visits, trajectories]")
        if any(len(set(row.tolist())) != len(row) for row in self.assignments):
            raise ValueError("a client is assigned twice during one visit")
        for path in self.assignments.T:
            if len(set(path.tolist())) != len(path):
                raise ValueError("a trajectory revisits a client within an outer round")


def build_without_replacement_schedule(
    client_ids: np.ndarray | list[int],
    num_trajectories: int,
    num_visits: int,
    rng: np.random.Generator,
) -> ClientPathSchedule:
    """Build the feasible schedule used by the corrected finite-population proof.

    A uniform client permutation and distinct cyclic offsets guarantee that each
    trajectory visits distinct clients and every visit uses distinct clients.
    """

    clients = np.asarray(client_ids, dtype=np.int64).reshape(-1)
    if clients.size == 0:
        raise ValueError("at least one training client is required")
    if len(np.unique(clients)) != len(clients):
        raise ValueError("client_ids must be unique")

    k_clients = len(clients)
    num_trajectories = int(num_trajectories)
    num_visits = int(num_visits)
    if not 1 <= num_trajectories <= k_clients:
        raise ValueError("num_trajectories must be in [1, number of clients]")
    if not 1 <= num_visits <= k_clients:
        raise ValueError("num_visits must be in [1, number of clients]")

    permutation = rng.permutation(clients)
    offsets = rng.choice(k_clients, size=num_trajectories, replace=False)
    visit_index = np.arange(num_visits, dtype=np.int64)[:, None]
    assignment_index = (visit_index + offsets[None, :]) % k_clients
    assignments = permutation[assignment_index]

    schedule = ClientPathSchedule(
        permutation=permutation,
        offsets=offsets,
        assignments=assignments,
    )
    schedule.validate()
    return schedule


def build_independent_schedule(
    client_ids: np.ndarray | list[int],
    num_trajectories: int,
    num_visits: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Published RADFed sampling, retained only as an experimental baseline."""

    clients = np.asarray(client_ids, dtype=np.int64).reshape(-1)
    if not 1 <= num_trajectories <= len(clients):
        raise ValueError("num_trajectories must be in [1, number of clients]")
    return np.stack(
        [rng.choice(clients, size=num_trajectories, replace=False) for _ in range(num_visits)],
        axis=0,
    )
