from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Iterable


# Observation channel names matching the exact order in ac_env.py:get_obs().
# Config: enable_sensors=True (11 rays), add_previous_obs_to_state=True (3 windows),
# use_target_speed=False, enable_task_id_in_obs=False.  Total = 125 dims.
def build_obs_channel_names(*, n_rays: int = 11, past_window: int = 3,
                            use_target_speed: bool = False) -> list[str]:
    base = [
        "speed", "gap", "last_ff", "rpm", "accel_x", "accel_y",
        "gear", "yaw_rate", "velocity_x", "velocity_y",
        "slip_fl", "slip_fr", "slip_rl", "slip_rr",
    ]
    if n_rays > 0:
        base += [f"ray_{i}" for i in range(n_rays)]

    names: list[str] = list(base)
    names.append("offtrack")
    names += [f"curvature_{i}" for i in range(12)]
    names += [f"prev_steer_{i}" for i in range(past_window)]
    names += [f"prev_throttle_{i}" for i in range(past_window)]
    names += [f"prev_brake_{i}" for i in range(past_window)]
    names += ["steer", "throttle", "brake"]
    # Previous observations (past_window copies of base channels)
    for w in range(past_window):
        names += [f"{ch}_t{w+1}" for ch in base]
    if use_target_speed:
        names += [f"target_speed_{i}" for i in range(12)]
    return names


OBS_CHANNEL_NAMES_DEFAULT = build_obs_channel_names()


PLAN_CODE_SCHEMA = {
    "speed_mode": ("conserve", "nominal", "push"),
    "brake_phase": ("early", "nominal", "late", "emergency"),
    "line_mode": ("tight", "neutral", "wide_exit", "recovery_line"),
    "stability_mode": ("neutral", "rotate", "stabilize"),
    "recovery_mode": ("off", "on"),
    "risk_mode": ("low", "medium", "high"),
}

DEFAULT_PLAN_CODE = {field_name: values[0] for field_name, values in PLAN_CODE_SCHEMA.items()}
DEFAULT_PLAN_CODE["speed_mode"] = "nominal"
DEFAULT_PLAN_CODE["brake_phase"] = "nominal"
DEFAULT_PLAN_CODE["line_mode"] = "neutral"
DEFAULT_PLAN_CODE["stability_mode"] = "neutral"


def canonicalize_plan_code(plan_code: dict[str, Any] | None) -> dict[str, str]:
    if not plan_code:
        return dict(DEFAULT_PLAN_CODE)

    canonical = {}
    for field_name, options in PLAN_CODE_SCHEMA.items():
        value = str(plan_code.get(field_name, DEFAULT_PLAN_CODE[field_name]))
        canonical[field_name] = value if value in options else DEFAULT_PLAN_CODE[field_name]
    return canonical


def plan_code_to_ids(plan_code: dict[str, Any] | None) -> dict[str, int]:
    normalized = canonicalize_plan_code(plan_code)
    return {
        field_name: PLAN_CODE_SCHEMA[field_name].index(value)
        for field_name, value in normalized.items()
    }


def plan_code_from_ids(plan_code_ids: dict[str, int] | None) -> dict[str, str]:
    if not plan_code_ids:
        return dict(DEFAULT_PLAN_CODE)
    decoded = {}
    for field_name, options in PLAN_CODE_SCHEMA.items():
        index = int(plan_code_ids.get(field_name, 0))
        decoded[field_name] = options[max(0, min(index, len(options) - 1))]
    return decoded


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def json_loads(payload: str | bytes | None, default: Any = None) -> Any:
    if payload is None:
        return default
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if payload == "":
        return default
    return json.loads(payload)


@dataclass
class UnifiedBackboneOutput:
    z_mid: list[float] = field(default_factory=list)
    plan_code: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PLAN_CODE))
    plan_code_ids: dict[str, int] = field(default_factory=dict)
    plan_logits: dict[str, list[float]] = field(default_factory=dict)
    value_hat: float = 0.0
    confidence: float = 0.0
    offtrack_prob: float = 0.0
    planner_version: str = "bootstrap"
    latency_ms: float = 0.0
    valid: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["plan_code"] = canonicalize_plan_code(payload["plan_code"])
        payload["plan_code_ids"] = plan_code_to_ids(payload["plan_code"])
        return payload


@dataclass
class SegmentRecord:
    run_id: str
    episode: int
    segment_id: str
    track_id: str
    car_id: str
    task_id: int
    summary_payload: dict[str, Any]
    raw_numeric_features: dict[str, float]
    frame_observations: list[list[float]]
    frame_event_bits: list[list[float]]
    planner_output: UnifiedBackboneOutput = field(default_factory=UnifiedBackboneOutput)
    future_progress_1s: float = 0.0
    future_progress_3s: float = 0.0
    future_return_3s: float = 0.0
    offtrack_next_3s: float = 0.0
    recovery_next_3s: float = 0.0
    lap_completed_next_10s: float = 0.0
    pseudo_labels: dict[str, Any] = field(default_factory=dict)
    label_confidence: float = 0.0
    evidence_segment_ids: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        planner_output = self.planner_output.to_dict()
        return {
            "run_id": self.run_id,
            "episode": int(self.episode),
            "segment_id": self.segment_id,
            "track_id": self.track_id,
            "car_id": self.car_id,
            "task_id": int(self.task_id),
            "summary_payload_json": json_dumps(self.summary_payload),
            "raw_numeric_features_json": json_dumps(self.raw_numeric_features),
            "frame_observations_json": json_dumps(self.frame_observations),
            "frame_event_bits_json": json_dumps(self.frame_event_bits),
            "planner_output_json": json_dumps(planner_output),
            "plan_code_json": json_dumps(planner_output["plan_code"]),
            "plan_code_ids_json": json_dumps(planner_output["plan_code_ids"]),
            "future_progress_1s": float(self.future_progress_1s),
            "future_progress_3s": float(self.future_progress_3s),
            "future_return_3s": float(self.future_return_3s),
            "offtrack_next_3s": float(self.offtrack_next_3s),
            "recovery_next_3s": float(self.recovery_next_3s),
            "lap_completed_next_10s": float(self.lap_completed_next_10s),
            "pseudo_labels_json": json_dumps(self.pseudo_labels),
            "label_confidence": float(self.label_confidence),
            "evidence_segment_ids_json": json_dumps(self.evidence_segment_ids),
        }


@dataclass
class CoachInteractionRecord:
    interaction_id: str
    run_id: str
    episode: int
    segment_ids: list[str]
    question: str
    answer: str
    plan_code_used: dict[str, str]
    evidence_segment_ids: list[str]
    player_feedback: str | None = None
    followed_flag: bool | None = None
    next_segment_delta: float | None = None
    usefulness_score: float | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "run_id": self.run_id,
            "episode": int(self.episode),
            "segment_ids": list(self.segment_ids),
            "question": self.question,
            "answer": self.answer,
            "plan_code_used": canonicalize_plan_code(self.plan_code_used),
            "evidence_segment_ids": list(self.evidence_segment_ids),
            "player_feedback": self.player_feedback,
            "followed_flag": self.followed_flag,
            "next_segment_delta": self.next_segment_delta,
            "usefulness_score": self.usefulness_score,
        }


@dataclass
class KnowledgeStoreRecord:
    chunk_id: str
    source_type: str
    track_id: str | None
    car_id: str | None
    topic_tags: list[str]
    text_chunk: str
    embedding: list[float] = field(default_factory=list)
    quality_score: float = 0.0
    source_reliability_score: float = 0.0

    def to_row(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_type": self.source_type,
            "track_id": self.track_id,
            "car_id": self.car_id,
            "topic_tags_json": json_dumps(self.topic_tags),
            "text_chunk": self.text_chunk,
            "embedding_json": json_dumps(self.embedding),
            "quality_score": float(self.quality_score),
            "source_reliability_score": float(self.source_reliability_score),
        }


def flatten_numeric_features(feature_dict: dict[str, float], ordered_keys: Iterable[str] | None = None) -> list[float]:
    keys = list(ordered_keys) if ordered_keys is not None else sorted(feature_dict.keys())
    return [float(feature_dict.get(key, 0.0)) for key in keys]
