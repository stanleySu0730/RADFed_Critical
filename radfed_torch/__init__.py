"""Python 3.12 / PyTorch implementation of the consolidated RADFed design."""

from .schedule import ClientPathSchedule, build_without_replacement_schedule
from .theory import (
    CriticalSchedule,
    ProfileStatistics,
    continuous_r_approximation,
    critical_learning_rate,
    critical_schedule,
    critical_schedule_candidates,
    drift_floor,
    fixed_r_critical_budget,
    fixed_r_critical_schedule,
    fixed_r_critical_schedule_candidates,
    population_correction,
    variance_coefficient,
)

__all__ = [
    "ClientPathSchedule",
    "CriticalSchedule",
    "ProfileStatistics",
    "build_without_replacement_schedule",
    "continuous_r_approximation",
    "critical_learning_rate",
    "critical_schedule",
    "critical_schedule_candidates",
    "drift_floor",
    "fixed_r_critical_budget",
    "fixed_r_critical_schedule",
    "fixed_r_critical_schedule_candidates",
    "population_correction",
    "variance_coefficient",
]
