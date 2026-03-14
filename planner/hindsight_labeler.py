"""Post-episode hindsight plan code relabeling using actual driving outcomes.

Instead of using heuristic-only labels (bootstrap problem), this module
refines plan code labels based on what actually happened after each segment:
future progress, offtrack events, speed outcomes, etc.
"""
from __future__ import annotations

import numpy as np

from .schemas import PLAN_CODE_SCHEMA, canonicalize_plan_code, plan_code_to_ids


# Ordered field names matching replay buffer hindsight_plan_ids columns
PLAN_FIELDS = list(PLAN_CODE_SCHEMA.keys())


def label_episode_hindsight(
    states: list[np.ndarray],
    rewards: list[float],
    dones: list[bool],
    env_infos: list[dict],
    *,
    gamma: float = 0.99,
    lookahead_steps: int = 75,  # ~3s at 25Hz
) -> list[dict[str, int]]:
    """Compute outcome-grounded plan code labels for an episode.

    Args:
        states: per-step observation vectors
        rewards: per-step scalar rewards
        dones: per-step termination flags
        env_infos: per-step env info dicts (contain speed, gap, offtrack, etc.)
        gamma: discount for computing future returns
        lookahead_steps: how far ahead to look for outcome grounding

    Returns:
        List of plan_code_id dicts (one per step), each mapping field name to int index.
    """
    n = len(states)
    if n == 0:
        return []

    # Compute discounted future returns for each step
    future_returns = np.zeros(n, dtype=np.float32)
    running = 0.0
    for t in reversed(range(n)):
        running = rewards[t] + gamma * running * (1.0 - float(dones[t]))
        future_returns[t] = running

    # Compute future offtrack flags (did the car go offtrack in the next lookahead_steps?)
    offtrack_ahead = np.zeros(n, dtype=np.float32)
    for t in range(n):
        end = min(t + lookahead_steps, n)
        for k in range(t, end):
            info = env_infos[k] if k < len(env_infos) else {}
            if info.get("out_of_track", False) or info.get("off_track", False):
                offtrack_ahead[t] = 1.0
                break

    # Compute speed trend over lookahead window
    speeds = np.array([
        _get_speed(env_infos[t]) if t < len(env_infos) else 0.0
        for t in range(n)
    ], dtype=np.float32)

    # Compute curvature at each step
    curvatures = np.array([
        _get_curvature(env_infos[t]) if t < len(env_infos) else 0.0
        for t in range(n)
    ], dtype=np.float32)

    # Compute gap magnitude
    gaps = np.array([
        abs(_get_gap(env_infos[t])) if t < len(env_infos) else 0.0
        for t in range(n)
    ], dtype=np.float32)

    # Percentile thresholds for return quality
    return_p25 = np.percentile(future_returns, 25) if n > 10 else future_returns.mean()
    return_p75 = np.percentile(future_returns, 75) if n > 10 else future_returns.mean()

    labels = []
    for t in range(n):
        plan_code = _label_step(
            future_return=future_returns[t],
            offtrack_ahead=offtrack_ahead[t],
            speed=speeds[t],
            curvature=curvatures[t],
            gap=gaps[t],
            return_p25=return_p25,
            return_p75=return_p75,
        )
        labels.append(plan_code_to_ids(plan_code))
    return labels


def _label_step(
    *,
    future_return: float,
    offtrack_ahead: float,
    speed: float,
    curvature: float,
    gap: float,
    return_p25: float,
    return_p75: float,
) -> dict[str, str]:
    """Label a single step using outcome data."""
    # speed_mode: based on outcome quality and curvature
    if offtrack_ahead > 0.5 or future_return < return_p25:
        speed_mode = "conserve"
    elif future_return > return_p75 and curvature < 0.05:
        speed_mode = "push"
    else:
        speed_mode = "nominal"

    # brake_phase: based on whether braking was needed (high curvature + outcome)
    if offtrack_ahead > 0.5 and curvature > 0.1:
        brake_phase = "emergency"
    elif curvature > 0.15 and future_return < return_p25:
        brake_phase = "early"
    elif curvature > 0.1 and future_return > return_p75:
        brake_phase = "late"
    else:
        brake_phase = "nominal"

    # line_mode: based on gap from reference line
    if gap > 1.3:
        line_mode = "recovery_line"
    elif gap > 0.7:
        line_mode = "tight"
    elif future_return > return_p75 and gap < 0.3:
        line_mode = "wide_exit"
    else:
        line_mode = "neutral"

    # stability_mode: based on gap and offtrack risk
    if offtrack_ahead > 0.5 or gap > 1.0:
        stability_mode = "stabilize"
    elif speed > 20.0 and curvature > 0.08 and future_return > return_p75:
        stability_mode = "rotate"
    else:
        stability_mode = "neutral"

    # recovery_mode: based on actual offtrack or large gap
    recovery_mode = "on" if (offtrack_ahead > 0.5 or gap > 1.3) else "off"

    # risk_mode: based on outcome quality
    if offtrack_ahead > 0.5 or future_return < return_p25:
        risk_mode = "high"
    elif curvature > 0.1 and speed > 25.0:
        risk_mode = "medium"
    else:
        risk_mode = "low"

    return canonicalize_plan_code({
        "speed_mode": speed_mode,
        "brake_phase": brake_phase,
        "line_mode": line_mode,
        "stability_mode": stability_mode,
        "recovery_mode": recovery_mode,
        "risk_mode": risk_mode,
    })


def plan_ids_to_array(plan_ids: dict[str, int]) -> np.ndarray:
    """Convert plan_code_ids dict to a fixed-order numpy array (6 ints)."""
    return np.array([plan_ids.get(f, 0) for f in PLAN_FIELDS], dtype=np.int64)


def _get_speed(info: dict) -> float:
    return float(info.get("speed", info.get("speed_ms", 0.0)))


def _get_curvature(info: dict) -> float:
    curv = info.get("curvature", info.get("curvature_ahead", None))
    if curv is not None:
        if isinstance(curv, (list, np.ndarray)):
            return float(np.mean(np.abs(curv[:3]))) if len(curv) > 0 else 0.0
        return abs(float(curv))
    return 0.0


def _get_gap(info: dict) -> float:
    return float(info.get("gap", info.get("gap_to_reference", 0.0)))
