from __future__ import annotations

from dataclasses import asdict, dataclass
import math


class NoFeasibleCriticalSchedule(ValueError):
    """The requested precision admits no implementable critical schedule."""


@dataclass(frozen=True)
class ProfileStatistics:
    """Empirical inputs used by finalized Equations (47), (49), and (56)."""

    sigma_sq: float
    g_sq: float
    l_smooth: float

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.l_smooth <= 0:
            raise ValueError("l_smooth must be positive")


@dataclass(frozen=True)
class CriticalSchedule:
    r_local_steps: int
    s_client_visits: int
    e_total_steps: int
    psi_var: float
    psi_drift: float
    error_floor: float
    precision_margin: float
    learning_rate: float
    round_seconds: float
    normalized_cost: float
    feasible_candidates: int
    continuous_e_critical: float | None = None


def population_correction(n_active: int, s_client_visits: int) -> float:
    """Return Phi(S) from finalized Equation (48)."""

    n_active = int(n_active)
    s_client_visits = int(s_client_visits)
    if n_active <= 1:
        raise ValueError("n_active must be greater than one")
    if not 1 <= s_client_visits <= n_active:
        raise ValueError("s_client_visits must be in [1, n_active]")
    return (n_active - s_client_visits) / (
        s_client_visits * (n_active - 1.0)
    )


def variance_coefficient(sigma_sq: float, n_active: int) -> float:
    """Return Psi_var from finalized Equation (47)."""

    sigma_sq = float(sigma_sq)
    n_active = int(n_active)
    if not math.isfinite(sigma_sq) or sigma_sq < 0:
        raise ValueError("sigma_sq must be finite and nonnegative")
    if n_active <= 1:
        raise ValueError("n_active must be greater than one")
    return sigma_sq * (math.sqrt(6.0) / (2.0 * n_active) + 0.5)


def drift_floor(g_sq: float, n_active: int, s_client_visits: int) -> float:
    """Return Psi_drift(S) from finalized Equation (49)."""

    g_sq = float(g_sq)
    if not math.isfinite(g_sq) or g_sq < 0:
        raise ValueError("g_sq must be finite and nonnegative")
    return g_sq * (population_correction(n_active, s_client_visits) + 1.0)


def critical_learning_rate(l_smooth: float, e_total_steps: int) -> float:
    """Return eta = 1/(sqrt(6) E L) from finalized Section 4.2."""

    l_smooth = float(l_smooth)
    e_total_steps = int(e_total_steps)
    if not math.isfinite(l_smooth) or l_smooth <= 0:
        raise ValueError("l_smooth must be positive and finite")
    if e_total_steps <= 0:
        raise ValueError("e_total_steps must be positive")
    return 1.0 / (math.sqrt(6.0) * e_total_steps * l_smooth)


def fixed_r_critical_budget(
    *,
    profile: ProfileStatistics,
    n_active: int,
    target_epsilon: float,
    r_local_steps: int,
) -> float:
    """Return the continuous E_crit(R) from finalized Equation (66)."""

    profile.validate()
    n_active = int(n_active)
    r_local_steps = int(r_local_steps)
    target_epsilon = float(target_epsilon)
    if n_active <= 1:
        raise ValueError("n_active must be greater than one")
    if r_local_steps <= 0:
        raise ValueError("r_local_steps must be positive")
    if not math.isfinite(target_epsilon) or target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive and finite")
    b_zero = (n_active - 2.0) * profile.g_sq / (n_active - 1.0)
    denominator = target_epsilon - b_zero
    if denominator <= 0:
        raise NoFeasibleCriticalSchedule(
            "target_epsilon must exceed the fixed drift floor b_0"
        )
    h_value = n_active * profile.g_sq / (n_active - 1.0)
    v_r = variance_coefficient(profile.sigma_sq, n_active) + h_value * r_local_steps
    return 2.0 * v_r / denominator


def fixed_r_critical_schedule_candidates(
    *,
    profile: ProfileStatistics,
    n_active: int,
    target_epsilon: float,
    r_local_steps: int,
    t_comm: float,
    t_comp: float,
    max_total_steps: int = 250,
) -> tuple[CriticalSchedule, ...]:
    """Return the feasible integer neighbours of Equation (66).

    Equation (66) gives a continuous local budget ``E_crit(R)`` for a fixed
    integer ``R``.  The algorithm additionally requires ``E = R*S`` and
    ``1 <= S <= N``.  Since the fixed-R cost decreases before ``E_crit`` and
    increases after it, only the floor and ceiling of ``E_crit/R`` need to be
    evaluated after applying the implementation limits.
    """

    profile.validate()
    n_active = int(n_active)
    r_local_steps = int(r_local_steps)
    max_total_steps = int(max_total_steps)
    t_comm = float(t_comm)
    t_comp = float(t_comp)
    if n_active <= 1:
        raise ValueError("n_active must be greater than one")
    if r_local_steps <= 0:
        raise ValueError("r_local_steps must be positive")
    if max_total_steps <= 0:
        raise ValueError("max_total_steps must be positive")
    if any(not math.isfinite(value) or value < 0 for value in (t_comm, t_comp)):
        raise ValueError("timing values must be finite and nonnegative")

    maximum_visits = min(n_active, max_total_steps // r_local_steps)
    if maximum_visits < 1:
        raise ValueError(
            "no fixed-R schedule fits max_total_steps; increase the total-step limit"
        )

    continuous_e = fixed_r_critical_budget(
        profile=profile,
        n_active=n_active,
        target_epsilon=target_epsilon,
        r_local_steps=r_local_steps,
    )
    continuous_s = continuous_e / r_local_steps
    neighbouring_visits = {
        max(1, min(maximum_visits, math.floor(continuous_s))),
        max(1, min(maximum_visits, math.ceil(continuous_s))),
    }

    psi_var = variance_coefficient(profile.sigma_sq, n_active)
    candidates: list[CriticalSchedule] = []
    for s_visits in sorted(neighbouring_visits):
        e_total = r_local_steps * s_visits
        psi_drift = drift_floor(profile.g_sq, n_active, s_visits)
        error_floor = psi_var / e_total + psi_drift
        precision_margin = float(target_epsilon) - error_floor
        if precision_margin <= 0:
            continue
        round_seconds = s_visits * t_comm + e_total * t_comp
        candidates.append(
            CriticalSchedule(
                r_local_steps=r_local_steps,
                s_client_visits=s_visits,
                e_total_steps=e_total,
                psi_var=psi_var,
                psi_drift=psi_drift,
                error_floor=error_floor,
                precision_margin=precision_margin,
                learning_rate=critical_learning_rate(profile.l_smooth, e_total),
                round_seconds=round_seconds,
                normalized_cost=round_seconds / precision_margin,
                feasible_candidates=0,
                continuous_e_critical=continuous_e,
            )
        )

    if not candidates:
        if maximum_visits < n_active:
            hint = "increase target_epsilon or max_total_steps"
        else:
            hint = (
                "increase target_epsilon; max_total_steps already permits "
                "all active-client visits"
            )
        raise NoFeasibleCriticalSchedule(
            f"no feasible fixed-R critical schedule; {hint}"
        )
    candidate_count = len(candidates)
    return tuple(
        CriticalSchedule(
            **{**asdict(candidate), "feasible_candidates": candidate_count}
        )
        for candidate in candidates
    )


def fixed_r_critical_schedule(
    *,
    profile: ProfileStatistics,
    n_active: int,
    target_epsilon: float,
    r_local_steps: int,
    t_comm: float,
    t_comp: float,
    max_total_steps: int = 250,
) -> CriticalSchedule:
    """Select the implementable fixed-R schedule implied by Equation (66)."""

    candidates = fixed_r_critical_schedule_candidates(
        profile=profile,
        n_active=n_active,
        target_epsilon=target_epsilon,
        r_local_steps=r_local_steps,
        t_comm=t_comm,
        t_comp=t_comp,
        max_total_steps=max_total_steps,
    )
    return min(
        candidates,
        key=lambda item: (
            item.normalized_cost,
            item.round_seconds,
            item.e_total_steps,
            item.s_client_visits,
        ),
    )


def continuous_r_approximation(
    *,
    profile: ProfileStatistics,
    n_active: int,
    t_comm: float,
    t_comp: float,
) -> float:
    """Return the continuous R approximation from finalized Equation (68)."""

    profile.validate()
    n_active = int(n_active)
    t_comm = float(t_comm)
    t_comp = float(t_comp)
    if n_active <= 1:
        raise ValueError("n_active must be greater than one")
    if t_comm <= 0 or t_comp <= 0:
        raise ValueError("t_comm and t_comp must be positive")
    if profile.g_sq <= 0:
        raise ValueError("g_sq must be positive for the R approximation")
    psi_var = variance_coefficient(profile.sigma_sq, n_active)
    return math.sqrt(
        t_comm * psi_var * (n_active - 1.0)
        / (t_comp * n_active * profile.g_sq)
    )


def critical_schedule_candidates(
    *,
    profile: ProfileStatistics,
    n_active: int,
    target_epsilon: float,
    t_comm: float,
    t_comp: float,
    max_total_steps: int = 250,
    max_local_steps: int = 25,
    restrict_r_to_one: bool = False,
) -> tuple[CriticalSchedule, ...]:
    """Enumerate the feasible integer schedules in finalized Equation (56).

    The common factor A = 6*sqrt(6)*Delta_0*L does not affect the argmin, so
    normalized_cost stores round_seconds / D(R,S).
    """

    profile.validate()
    n_active = int(n_active)
    target_epsilon = float(target_epsilon)
    t_comm = float(t_comm)
    t_comp = float(t_comp)
    max_total_steps = int(max_total_steps)
    max_local_steps = int(max_local_steps)
    if n_active <= 1:
        raise ValueError("n_active must be greater than one")
    if not math.isfinite(target_epsilon) or target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive and finite")
    if any(not math.isfinite(value) or value < 0 for value in (t_comm, t_comp)):
        raise ValueError("timing values must be finite and nonnegative")
    if max_total_steps <= 0 or max_local_steps <= 0:
        raise ValueError("search limits must be positive")

    psi_var = variance_coefficient(profile.sigma_sq, n_active)
    candidates: list[CriticalSchedule] = []
    r_values = (1,) if restrict_r_to_one else range(1, max_local_steps + 1)
    for s_visits in range(1, n_active + 1):
        psi_drift = drift_floor(profile.g_sq, n_active, s_visits)
        for r_steps in r_values:
            e_total = r_steps * s_visits
            if e_total > max_total_steps:
                continue
            error_floor = psi_var / e_total + psi_drift
            precision_margin = target_epsilon - error_floor
            if precision_margin <= 0:
                continue
            round_seconds = s_visits * t_comm + e_total * t_comp
            normalized_cost = round_seconds / precision_margin
            candidates.append(
                CriticalSchedule(
                    r_local_steps=r_steps,
                    s_client_visits=s_visits,
                    e_total_steps=e_total,
                    psi_var=psi_var,
                    psi_drift=psi_drift,
                    error_floor=error_floor,
                    precision_margin=precision_margin,
                    learning_rate=critical_learning_rate(
                        profile.l_smooth, e_total
                    ),
                    round_seconds=round_seconds,
                    normalized_cost=normalized_cost,
                    feasible_candidates=0,
                )
            )

    if not candidates:
        raise NoFeasibleCriticalSchedule(
            "no feasible critical schedule; increase epsilon or the search limits"
        )
    candidate_count = len(candidates)
    return tuple(
        CriticalSchedule(
            **{**asdict(candidate), "feasible_candidates": candidate_count}
        )
        for candidate in candidates
    )


def critical_schedule(
    *,
    profile: ProfileStatistics,
    n_active: int,
    target_epsilon: float,
    t_comm: float,
    t_comp: float,
    max_total_steps: int = 250,
    max_local_steps: int = 25,
    restrict_r_to_one: bool = False,
) -> CriticalSchedule:
    """Return the minimizing feasible integer schedule from Equation (56)."""

    candidates = critical_schedule_candidates(
        profile=profile,
        n_active=n_active,
        target_epsilon=target_epsilon,
        t_comm=t_comm,
        t_comp=t_comp,
        max_total_steps=max_total_steps,
        max_local_steps=max_local_steps,
        restrict_r_to_one=restrict_r_to_one,
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
