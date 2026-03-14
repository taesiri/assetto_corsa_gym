from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schemas import SegmentRecord, UnifiedBackboneOutput, canonicalize_plan_code


DEFAULT_OBS_COLUMNS = [
    "speed",
    "gap",
    "LastFF",
    "RPM",
    "accelX",
    "accelY",
    "actualGear",
    "angular_velocity_y",
    "local_velocity_x",
    "local_velocity_y",
    "SlipAngle_fl",
    "SlipAngle_fr",
    "SlipAngle_rl",
    "SlipAngle_rr",
]

EVENT_FLAG_ORDER = [
    "corner_entry",
    "overspeed",
    "gap_alert",
    "heading_alert",
    "recovery",
    "off_track_recent",
]


def _column_or_zeros(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.zeros(len(frame), dtype=np.float32)
    return frame[column].fillna(0.0).to_numpy(dtype=np.float32)


def _safe_tail_trend(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(values[-1] - values[0])


def _list_column(value: Any, expected_len: int, fallback_value: float = 0.0) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        vector = [float(v) for v in value[:expected_len]]
        if len(vector) < expected_len:
            vector.extend([fallback_value] * (expected_len - len(vector)))
        return vector
    return [float(fallback_value)] * expected_len


def build_event_bits(segment: pd.DataFrame) -> list[list[float]]:
    curvature = _column_or_zeros(segment, "curvature_ahead_ratio")
    overspeed = _column_or_zeros(segment, "overspeed_error")
    gap = np.abs(_column_or_zeros(segment, "gap"))
    heading = np.abs(_column_or_zeros(segment, "heading_error"))
    out_of_track = _column_or_zeros(segment, "out_of_track")
    lateral_speed = np.abs(_column_or_zeros(segment, "local_velocity_y"))

    rows = []
    off_track_recent = 0.0
    for idx in range(len(segment)):
        off_track_recent = max(off_track_recent * 0.85, float(out_of_track[idx] > 0.5))
        rows.append(
            [
                float(curvature[idx] > 0.22 and _column_or_zeros(segment, "speed")[idx] > 10.0),
                float(overspeed[idx] > 3.0),
                float(gap[idx] > 1.0),
                float(heading[idx] > 0.12),
                float(out_of_track[idx] > 0.5 or gap[idx] > 1.4 or lateral_speed[idx] > 4.0),
                float(off_track_recent > 0.1),
            ]
        )
    return rows


def build_frame_observations(segment: pd.DataFrame, obs_columns: list[str] | None = None) -> list[list[float]]:
    obs_columns = obs_columns or DEFAULT_OBS_COLUMNS
    vectors = []
    for _, row in segment.iterrows():
        vectors.append([float(row.get(column, 0.0)) for column in obs_columns])
    return vectors


def aggregate_numeric_features(segment: pd.DataFrame) -> dict[str, float]:
    speed = _column_or_zeros(segment, "speed")
    gap = np.abs(_column_or_zeros(segment, "gap"))
    heading = np.abs(_column_or_zeros(segment, "heading_error"))
    local_vy = np.abs(_column_or_zeros(segment, "local_velocity_y"))
    ang_y = np.abs(_column_or_zeros(segment, "angular_velocity_y"))
    overspeed = _column_or_zeros(segment, "overspeed_error")
    throttle = _column_or_zeros(segment, "accStatus")
    brake = _column_or_zeros(segment, "brakeStatus")
    steer = _column_or_zeros(segment, "steerAngle")
    curvature = _column_or_zeros(segment, "curvature_ahead_ratio")
    target_speed = _column_or_zeros(segment, "heuristic_target_speed")
    out_of_track = _column_or_zeros(segment, "out_of_track")

    return {
        "speed_mean": float(speed.mean()) if len(speed) else 0.0,
        "speed_max": float(speed.max()) if len(speed) else 0.0,
        "speed_trend": _safe_tail_trend(speed),
        "gap_abs_mean": float(gap.mean()) if len(gap) else 0.0,
        "gap_abs_max": float(gap.max()) if len(gap) else 0.0,
        "heading_abs_mean": float(heading.mean()) if len(heading) else 0.0,
        "heading_abs_max": float(heading.max()) if len(heading) else 0.0,
        "local_velocity_y_abs_mean": float(local_vy.mean()) if len(local_vy) else 0.0,
        "local_velocity_y_abs_max": float(local_vy.max()) if len(local_vy) else 0.0,
        "angular_velocity_y_abs_mean": float(ang_y.mean()) if len(ang_y) else 0.0,
        "overspeed_mean": float(overspeed.mean()) if len(overspeed) else 0.0,
        "overspeed_max": float(overspeed.max()) if len(overspeed) else 0.0,
        "throttle_mean": float(throttle.mean()) if len(throttle) else 0.0,
        "brake_mean": float(brake.mean()) if len(brake) else 0.0,
        "steer_mean": float(steer.mean()) if len(steer) else 0.0,
        "steer_std": float(steer.std()) if len(steer) else 0.0,
        "curvature_mean": float(curvature.mean()) if len(curvature) else 0.0,
        "curvature_max": float(curvature.max()) if len(curvature) else 0.0,
        "target_speed_mean": float(target_speed.mean()) if len(target_speed) else 0.0,
        "target_speed_min": float(target_speed.min()) if len(target_speed) else 0.0,
        "out_of_track_frac": float(out_of_track.mean()) if len(out_of_track) else 0.0,
    }


def build_summary_payload(
    segment: pd.DataFrame,
    track_id: str,
    car_id: str,
    task_id: int,
    prior_outcomes: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    prior_outcomes = prior_outcomes or []
    speed = _column_or_zeros(segment, "speed")
    throttle = _column_or_zeros(segment, "accStatus")
    brake = _column_or_zeros(segment, "brakeStatus")
    steer = _column_or_zeros(segment, "steerAngle")
    gap = _column_or_zeros(segment, "gap")
    heading = _column_or_zeros(segment, "heading_error")
    slip = np.maximum.reduce(
        [
            np.abs(_column_or_zeros(segment, "SlipAngle_fl")),
            np.abs(_column_or_zeros(segment, "SlipAngle_fr")),
            np.abs(_column_or_zeros(segment, "SlipAngle_rl")),
            np.abs(_column_or_zeros(segment, "SlipAngle_rr")),
        ]
    )
    yaw_rate = _column_or_zeros(segment, "angular_velocity_y")
    curvature_bins = _list_column(segment.iloc[-1].get("curvature_lookahead"), 12, 0.0)
    target_speed_bins = _list_column(segment.iloc[-1].get("target_speed_lookahead"), 12, float(speed.mean() if len(speed) else 0.0))
    progress = float(segment.iloc[-1].get("progress_laps", 0.0))

    return {
        "track_id": track_id,
        "car_id": car_id,
        "task_id": int(task_id),
        "lap_progress_bin": int(np.clip(progress * 20.0, 0.0, 19.0)),
        "speed_mean": float(speed.mean()) if len(speed) else 0.0,
        "speed_max": float(speed.max()) if len(speed) else 0.0,
        "speed_trend": _safe_tail_trend(speed),
        "throttle_mean": float(throttle.mean()) if len(throttle) else 0.0,
        "brake_mean": float(brake.mean()) if len(brake) else 0.0,
        "steer_mean": float(steer.mean()) if len(steer) else 0.0,
        "steer_std": float(steer.std()) if len(steer) else 0.0,
        "gap_mean": float(gap.mean()) if len(gap) else 0.0,
        "gap_abs_max": float(np.abs(gap).max()) if len(gap) else 0.0,
        "heading_error_mean": float(heading.mean()) if len(heading) else 0.0,
        "heading_error_abs_max": float(np.abs(heading).max()) if len(heading) else 0.0,
        "slip_abs_max": float(slip.max()) if len(slip) else 0.0,
        "yaw_rate_mean": float(yaw_rate.mean()) if len(yaw_rate) else 0.0,
        "overspeed_max": float(_column_or_zeros(segment, "overspeed_error").max()) if len(segment) else 0.0,
        "curvature_bins": curvature_bins,
        "target_speed_bins": target_speed_bins,
        "prior_outcomes": prior_outcomes[-4:],
        "event_flags": {
            "corner_entry": bool(np.max(_column_or_zeros(segment, "curvature_ahead_ratio")) > 0.22),
            "corner_exit": bool(segment.iloc[-1].get("curvature_ahead_ratio", 0.0) < 0.15),
            "recovery": bool(np.max(np.abs(gap)) > 1.25 or np.max(_column_or_zeros(segment, "out_of_track")) > 0.5),
            "off_track_recent": bool(np.max(_column_or_zeros(segment, "out_of_track")) > 0.5),
        },
    }


def infer_plan_code(summary_payload: dict[str, Any], features: dict[str, float]) -> tuple[dict[str, str], dict[str, Any], float]:
    speed_mode = "nominal"
    if features["speed_mean"] < max(summary_payload["target_speed_bins"][0] * 0.7, 8.0):
        speed_mode = "conserve"
    elif features["overspeed_max"] > 4.0 or features["speed_max"] > max(summary_payload["target_speed_bins"]) * 1.12:
        speed_mode = "push"

    brake_phase = "nominal"
    if features["overspeed_max"] > 5.5:
        brake_phase = "emergency"
    elif features["brake_mean"] > 0.35 and features["speed_trend"] < -1.0:
        brake_phase = "early"
    elif features["overspeed_mean"] > 1.0 and features["brake_mean"] < 0.15:
        brake_phase = "late"

    line_mode = "neutral"
    if features["gap_abs_max"] > 1.3:
        line_mode = "recovery_line"
    elif features["gap_abs_mean"] > 0.65:
        line_mode = "tight"
    elif features["speed_trend"] > 1.0 and features["heading_abs_max"] < 0.08:
        line_mode = "wide_exit"

    stability_mode = "neutral"
    if features["local_velocity_y_abs_max"] > 4.0 or features["angular_velocity_y_abs_mean"] > 0.9:
        stability_mode = "stabilize"
    elif features["speed_mean"] > 14.0 and features["heading_abs_mean"] < 0.05:
        stability_mode = "rotate"

    recovery_mode = "on" if (
        summary_payload["event_flags"]["recovery"] or features["out_of_track_frac"] > 0.0 or features["gap_abs_max"] > 1.3
    ) else "off"

    risk_mode = "low"
    if features["overspeed_max"] > 4.0 or features["heading_abs_max"] > 0.12:
        risk_mode = "medium"
    if features["overspeed_max"] > 6.0 or features["gap_abs_max"] > 1.5 or features["out_of_track_frac"] > 0.0:
        risk_mode = "high"

    plan_code = canonicalize_plan_code(
        {
            "speed_mode": speed_mode,
            "brake_phase": brake_phase,
            "line_mode": line_mode,
            "stability_mode": stability_mode,
            "recovery_mode": recovery_mode,
            "risk_mode": risk_mode,
        }
    )
    pseudo_labels = {
        "too_fast_for_corner": bool(features["overspeed_max"] > 3.0),
        "late_brake": bool(brake_phase in ("late", "emergency")),
        "early_brake": bool(brake_phase == "early"),
        "unstable_entry": bool(features["heading_abs_max"] > 0.12 or features["local_velocity_y_abs_max"] > 4.0),
        "good_exit": bool(features["speed_trend"] > 1.0 and features["gap_abs_max"] < 0.7),
        "recovery": bool(recovery_mode == "on"),
        "line_drift": bool(features["gap_abs_mean"] > 0.8),
    }
    confidence = float(np.clip(1.0 - (0.15 * sum(bool(v) for v in pseudo_labels.values() if v)), 0.15, 0.95))
    return plan_code, pseudo_labels, confidence


def compute_future_targets(segment: pd.DataFrame, full_episode: pd.DataFrame, end_index: int, ctrl_hz: int = 25) -> dict[str, float]:
    progress_now = float(segment.iloc[-1].get("lap_progress_distance", segment.iloc[-1].get("LapDist", 0.0)))
    reward_now = float(segment.iloc[-1].get("reward", 0.0))
    window_1s = min(len(full_episode) - 1, end_index + ctrl_hz)
    window_3s = min(len(full_episode) - 1, end_index + 3 * ctrl_hz)
    window_10s = min(len(full_episode) - 1, end_index + 10 * ctrl_hz)

    progress_1s = float(full_episode.iloc[window_1s].get("lap_progress_distance", full_episode.iloc[window_1s].get("LapDist", 0.0)))
    progress_3s = float(full_episode.iloc[window_3s].get("lap_progress_distance", full_episode.iloc[window_3s].get("LapDist", 0.0)))
    reward_3s = float(full_episode.iloc[end_index: window_3s + 1].get("reward", pd.Series(dtype=float)).sum())
    offtrack_next_3s = float(full_episode.iloc[end_index: window_3s + 1].get("out_of_track", pd.Series(dtype=float)).max())
    gap_window = np.abs(full_episode.iloc[end_index: window_3s + 1].get("gap", pd.Series(dtype=float)).fillna(0.0).to_numpy(dtype=np.float32))
    recovery_next_3s = float(
        (float(np.max(gap_window)) if gap_window.size else 0.0) > 1.0
    )
    lap_counts = full_episode.iloc[end_index: window_10s + 1].get("LapCount", pd.Series(dtype=float)).to_numpy(dtype=np.float32)
    max_lap_count = float(np.max(lap_counts)) if lap_counts.size else float(segment.iloc[-1].get("LapCount", 0.0))
    lap_completed_next_10s = float(
        max_lap_count > float(segment.iloc[-1].get("LapCount", 0.0))
    )
    return {
        "future_progress_1s": progress_1s - progress_now,
        "future_progress_3s": progress_3s - progress_now,
        "future_return_3s": reward_3s - reward_now,
        "offtrack_next_3s": offtrack_next_3s,
        "recovery_next_3s": recovery_next_3s,
        "lap_completed_next_10s": lap_completed_next_10s,
    }


def segment_episode_dataframe(
    episode_df: pd.DataFrame,
    *,
    run_id: str,
    episode: int,
    track_id: str,
    car_id: str,
    task_id: int,
    segment_steps: int,
    obs_columns: list[str] | None = None,
) -> list[SegmentRecord]:
    records: list[SegmentRecord] = []
    prior_outcomes: list[dict[str, float]] = []
    obs_columns = obs_columns or DEFAULT_OBS_COLUMNS

    for segment_start in range(0, len(episode_df), segment_steps):
        segment_end = min(len(episode_df), segment_start + segment_steps)
        segment = episode_df.iloc[segment_start:segment_end].reset_index(drop=True)
        if len(segment) < max(4, segment_steps // 2):
            continue

        features = aggregate_numeric_features(segment)
        summary_payload = build_summary_payload(segment, track_id=track_id, car_id=car_id, task_id=task_id, prior_outcomes=prior_outcomes)
        plan_code, pseudo_labels, label_confidence = infer_plan_code(summary_payload, features)
        future_targets = compute_future_targets(segment, episode_df, segment_end - 1)
        segment_id = f"{run_id}-ep{episode:04d}-seg{len(records):05d}"
        planner_output = UnifiedBackboneOutput(
            z_mid=[],
            plan_code=plan_code,
            value_hat=float(future_targets["future_progress_3s"]),
            confidence=label_confidence,
            offtrack_prob=float(future_targets["offtrack_next_3s"]),
            planner_version="bootstrap",
            latency_ms=0.0,
            valid=False,
        )
        records.append(
            SegmentRecord(
                run_id=run_id,
                episode=episode,
                segment_id=segment_id,
                track_id=track_id,
                car_id=car_id,
                task_id=task_id,
                summary_payload=summary_payload,
                raw_numeric_features=features,
                frame_observations=build_frame_observations(segment, obs_columns=obs_columns),
                frame_event_bits=build_event_bits(segment),
                planner_output=planner_output,
                pseudo_labels=pseudo_labels,
                label_confidence=label_confidence,
                evidence_segment_ids=[],
                **future_targets,
            )
        )
        prior_outcomes.append(
            {
                "future_progress_3s": future_targets["future_progress_3s"],
                "future_return_3s": future_targets["future_return_3s"],
                "offtrack_next_3s": future_targets["offtrack_next_3s"],
            }
        )
    return records


def infer_track_and_car_from_dataframe(frame: pd.DataFrame, default_track: str, default_car: str) -> tuple[str, str]:
    track = str(frame["track"].iloc[0]) if "track" in frame.columns and len(frame) else default_track
    car = str(frame["car"].iloc[0]) if "car" in frame.columns and len(frame) else default_car
    return track, car


def build_segment_dataframe(
    lap_files: list[Path],
    *,
    output_dir: Path,
    default_track: str,
    default_car: str,
    task_index: dict[tuple[str, str], int],
    segment_steps: int = 25,
    obs_columns: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lap_file in lap_files:
        frame = pd.read_parquet(lap_file)
        if frame.empty:
            continue
        track_id, car_id = infer_track_and_car_from_dataframe(frame, default_track=default_track, default_car=default_car)
        task_id = int(task_index.get((track_id, car_id), 0))
        run_id = lap_file.parent.parent.name
        episodes = frame.groupby("episode") if "episode" in frame.columns else [(0, frame)]
        for episode, episode_df in episodes:
            segment_records = segment_episode_dataframe(
                episode_df.reset_index(drop=True),
                run_id=run_id,
                episode=int(episode),
                track_id=track_id,
                car_id=car_id,
                task_id=task_id,
                segment_steps=segment_steps,
                obs_columns=obs_columns,
            )
            rows.extend(record.to_row() for record in segment_records)
    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        df.to_parquet(output_dir / "segments.parquet", index=False)
        df.to_csv(output_dir / "segments.csv", index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified-backbone segment parquet shards from lap parquet data.")
    parser.add_argument("--lap-root", required=True, help="Root directory containing lap parquet files.")
    parser.add_argument("--output-dir", required=True, help="Directory to write segment parquet shards into.")
    parser.add_argument("--track", default="monza", help="Default track id when the lap data does not contain one.")
    parser.add_argument("--car", default="ks_mazda_miata", help="Default car id when the lap data does not contain one.")
    parser.add_argument("--task-track", action="append", dest="task_tracks", default=None, help="Task track list override.")
    parser.add_argument("--task-car", action="append", dest="task_cars", default=None, help="Task car list override.")
    parser.add_argument("--segment-steps", type=int, default=25)
    parser.add_argument("--limit-files", type=int, default=0, help="Optional cap on the number of lap parquet files to process.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lap_root = Path(args.lap_root)
    output_dir = Path(args.output_dir)
    lap_files = sorted(lap_root.rglob("*.parquet"))
    if args.limit_files > 0:
        lap_files = lap_files[: args.limit_files]

    task_tracks = args.task_tracks or [args.track]
    task_cars = args.task_cars or [args.car]
    task_index = {}
    index = 0
    for track in task_tracks:
        for car in task_cars:
            task_index[(track, car)] = index
            index += 1

    df = build_segment_dataframe(
        lap_files,
        output_dir=output_dir,
        default_track=args.track,
        default_car=args.car,
        task_index=task_index,
        segment_steps=args.segment_steps,
    )
    print(f"Built {len(df)} segments from {len(lap_files)} lap files into {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
