from __future__ import annotations

import argparse
import json
import math
import re
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import yaml

try:
    from tensorboard.backend.event_processing import event_accumulator
except Exception:  # pragma: no cover - dashboard can run without tensorboard fallback
    event_accumulator = None


MAX_RUNS = 24
RECENT_EPISODES = 25
LOG_TAIL_LINES = 60
PROCESS_PATTERNS = (
    "train_for_duration.py",
    "train.py",
    "live_experiment_sweep.py",
    "dashboard.app",
    "Start-UnifiedDashboard.ps1",
)

ARCHITECTURE = {
    "title": "Unified Qwen Shared-Backbone Driving + Coaching Stack v4.1",
    "status": {
        "runtime": "implemented",
        "dashboard": "implemented",
        "offline_pretrain": "implemented",
        "live_shared_run": "tested",
        "true_qwen_runtime": "blocked",
    },
    "components": [
        "StateTokenizer + TemporalCompressor in planner/unified_backbone.py",
        "SharedBackboneRuntime mid-layer cache in planner/unified_backbone.py",
        "SharedBackboneSAC actor/Q heads in algorithm/discor/discor/algorithm/shared_backbone_sac.py",
        "Grounded coach API in coach/api.py",
        "SegmentRecord and related schemas in planner/schemas.py",
    ],
    "phases": [
        "Phase 0: uv migration and CUDA PyTorch under uv",
        "Phase 1: segment store build from curated lap packs",
        "Phase 2: offline backbone/value pretraining",
        "Phase 3: short live shared-backbone profile run",
        "Phase 4: true Qwen runtime enablement",
    ],
    "blockers": [
        "Current Python 3.9-compatible transformers build does not load model_type qwen3_5, so Qwen 3.5 checkpoints fall back.",
        "Offline segment pretrain uses compact telemetry while the live env state is 125-D, so pretrain and live state spaces are not yet aligned.",
        "Live behavior is still dominated by launch-assist and low-speed termination.",
    ],
    "docs": [
        "C:/Workspace/RacingSim/docs/unified_backbone_dashboard_status_2026-03-13.md",
        "C:/Workspace/RacingSim/docs/handoff_march_12.md",
        "C:/Workspace/RacingSim/docs/SETUP_AND_TRAINING.md",
    ],
}


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_for_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item") and callable(value.item):
        try:
            return sanitize_for_json(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist") and not isinstance(value, (bytes, bytearray)):
        try:
            return sanitize_for_json(value.tolist())
        except Exception:
            pass
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def json_load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def tail_lines(path: Path, limit: int = LOG_TAIL_LINES) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    except Exception:
        return []


def probe_tcp(host: str, port: int, timeout_s: float = 0.75) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def local_ipv4_addresses() -> list[str]:
    ips = {"127.0.0.1"}
    for _, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if getattr(addr, "family", None) == socket.AF_INET and addr.address:
                ips.add(addr.address)
    return sorted(ips)


def detect_run_kind(run_dir: Path) -> str | None:
    if (run_dir / "summary.csv").exists():
        return "live_run"
    if (run_dir / "offline_metrics.json").exists():
        return "offline_pretrain"
    if (run_dir / "segments.parquet").exists() or (run_dir / "segments.csv").exists():
        return "segment_store"
    if (run_dir / "status.json").exists():
        return "sweep"
    return None


def numeric_series(frame: pd.DataFrame, column: str) -> list[float]:
    if column not in frame.columns:
        return []
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return [float(value) for value in values.tail(RECENT_EPISODES).tolist()]


def load_scalar_frame(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "live_scalars.csv"
    if csv_path.exists():
        try:
            frame = pd.read_csv(csv_path)
            if {"tag", "step", "value"}.issubset(frame.columns):
                return frame
        except Exception:
            pass

    if event_accumulator is None:
        return pd.DataFrame(columns=["tag", "step", "value"])

    summary_dir = run_dir / "summary"
    event_files = sorted(summary_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return pd.DataFrame(columns=["tag", "step", "value"])

    try:
        accumulator = event_accumulator.EventAccumulator(
            str(summary_dir),
            size_guidance={event_accumulator.SCALARS: 0},
        )
        accumulator.Reload()
        records: list[dict[str, Any]] = []
        for tag in accumulator.Tags().get("scalars", []):
            for item in accumulator.Scalars(tag):
                records.append({"tag": tag, "step": int(item.step), "value": float(item.value)})
        return pd.DataFrame.from_records(records, columns=["tag", "step", "value"])
    except Exception:
        return pd.DataFrame(columns=["tag", "step", "value"])


def scalar_tail(frame: pd.DataFrame, tag: str, limit: int = 240) -> dict[str, list[float]]:
    if frame.empty:
        return {"steps": [], "values": []}
    tagged = frame.loc[frame["tag"] == tag, ["step", "value"]].dropna()
    if tagged.empty:
        return {"steps": [], "values": []}
    tagged = tagged.tail(limit)
    return {
        "steps": [int(step) for step in tagged["step"].tolist()],
        "values": [float(value) for value in tagged["value"].tolist()],
    }


def latest_scalar_values(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}
    tagged = frame.dropna(subset=["tag", "step", "value"]).sort_values("step")
    latest = tagged.groupby("tag", sort=False).tail(1)
    return {str(row["tag"]): float(row["value"]) for _, row in latest.iterrows()}


def parse_logged_config(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {}
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return {}

    start_index = None
    for index, line in enumerate(lines[:400]):
        if "Configuration:" in line:
            start_index = index + 1
            break
    if start_index is None:
        return {}

    config_lines: list[str] = []
    yaml_like = re.compile(r"^\s*([A-Za-z0-9_.-]+:\s*.*|- .*)?$")
    prefix = re.compile(r"^\d{4}-\d{2}-\d{2} .*?\[[A-Z]+\] [^ ]+ (.*)$")

    for raw in lines[start_index:start_index + 260]:
        match = prefix.match(raw)
        content = match.group(1) if match else raw
        if not yaml_like.match(content):
            break
        if re.match(r"^[A-Za-z]:\\", content.strip()):
            continue
        config_lines.append(content)

    if not config_lines:
        return {}

    try:
        parsed = yaml.safe_load("\n".join(config_lines))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def select_hyperparameters(config_snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    section_keys = {
        "Runtime": ["cuda_device_index", "low_fi_training", "profile_mode"],
        "UnifiedBackbone": [
            "enabled",
            "model_name",
            "fallback_model_name",
            "freeze_backbone",
            "cache_refresh_hz",
            "branch_layer_4b",
            "branch_layer_2b",
            "control_delta_dim",
        ],
        "Agent": [
            "batch_size",
            "memory_size",
            "offline_buffer_size",
            "update_every_steps",
            "update_steps_per_call",
            "post_episode_update_budget_s",
            "start_steps",
            "checkpoint_freq",
        ],
        "SAC": ["gamma", "nstep", "policy_lr", "q_lr", "entropy_lr"],
        "AssettoCorsa": [
            "ego_sampling_freq",
            "track",
            "car",
            "task_tracks",
            "task_cars",
            "heldout_track",
            "use_unified_backbone_guidance",
            "use_action_prior_sampling",
            "use_teacher_controller",
        ],
    }
    for section, keys in section_keys.items():
        section_payload = config_snapshot.get(section, {})
        if not isinstance(section_payload, dict):
            continue
        values = {key: section_payload.get(key) for key in keys if key in section_payload}
        if values:
            selected[section] = values
    return selected


def summarize_training_flow(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "episode": [],
            "reset_time_s": [],
            "live_time_s": [],
            "wall_time_s": [],
            "env_steps_per_sec": [],
            "gradient_updates_per_sec": [],
            "planner_confidence_mean": [],
            "planner_latency_p95_ms": [],
            "cache_refresh_count": [],
            "recent_rows": [],
        }

    reduced_cols = [
        "ep_count",
        "reset_time_s",
        "episode_live_time_s",
        "episode_wall_time_s",
        "env_steps_per_sec",
        "gradient_updates_per_sec",
        "planner_confidence_mean",
        "planner_latency_p95_ms",
        "cache_refresh_count",
        "packages_lost",
        "speed_mean",
        "lap_progress_laps",
    ]
    available_cols = [column for column in reduced_cols if column in frame.columns]
    return {
        "episode": numeric_series(frame, "ep_count"),
        "reset_time_s": numeric_series(frame, "reset_time_s"),
        "live_time_s": numeric_series(frame, "episode_live_time_s"),
        "wall_time_s": numeric_series(frame, "episode_wall_time_s"),
        "env_steps_per_sec": numeric_series(frame, "env_steps_per_sec"),
        "gradient_updates_per_sec": numeric_series(frame, "gradient_updates_per_sec"),
        "planner_confidence_mean": numeric_series(frame, "planner_confidence_mean"),
        "planner_latency_p95_ms": numeric_series(frame, "planner_latency_p95_ms"),
        "cache_refresh_count": numeric_series(frame, "cache_refresh_count"),
        "recent_rows": frame[available_cols].tail(RECENT_EPISODES).fillna("").to_dict(orient="records"),
    }


def compact_segment_sample(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    row = frame.head(1).iloc[0]
    keys = [
        "segment_id",
        "track_id",
        "car_id",
        "task_id",
        "future_progress_3s",
        "future_return_3s",
        "offtrack_next_3s",
        "plan_code_json",
        "label_confidence",
    ]
    return {
        key: (row[key] if key in row.index else None)
        for key in keys
    }


def language_preview(run: dict[str, Any]) -> list[str]:
    if run["kind"] == "live_run":
        lines = []
        if float(run.get("latest_speed_mean", 0.0)) < 1.0:
            lines.append("The controller is still trapped in a low-speed launch or recovery regime.")
        else:
            lines.append("The controller is moving, but it is not yet converting rollout time into useful lap progress.")
        if float(run.get("latest_packages_lost", 0.0)) > 1000:
            lines.append("Packet loss remains high enough to distort control quality.")
        if run.get("planner_model_name", "unknown") == "fallback":
            lines.append("The live backbone is using the internal fallback encoder instead of Qwen.")
        lines.append("Next action: fix true Qwen loading or deliberately train the fallback encoder as a separate baseline.")
        return lines
    if run["kind"] == "offline_pretrain":
        return [
            f"Offline pretraining completed {run.get('epochs', 0)} epochs.",
            f"Latest loss is {run.get('latest_loss', 0.0):.4f}.",
            "This is a bootstrap artifact, not yet proof of live driving improvement.",
        ]
    if run["kind"] == "segment_store":
        return [
            f"This segment store contains {run.get('segments', 0)} segments.",
            "It is the current offline training source of truth.",
        ]
    return ["No grounded language preview for this artifact."]


def summarize_live_run(run_dir: Path) -> dict[str, Any]:
    frame = pd.read_csv(run_dir / "summary.csv")
    frame = frame.sort_values("ep_count") if "ep_count" in frame.columns else frame
    latest = frame.tail(1).iloc[0].to_dict() if not frame.empty else {}
    profile = json_load(run_dir / "profile_summary.json", {})
    scalar_frame = load_scalar_frame(run_dir)
    scalar_latest = latest_scalar_values(scalar_frame)
    scalar_tags = sorted(scalar_frame["tag"].dropna().unique().tolist()) if not scalar_frame.empty else []
    scalar_series = {
        tag: scalar_tail(scalar_frame, tag)
        for tag in [
            "loss/policy",
            "loss/Q",
            "loss/entropy",
            "stats/alpha",
            "stats/entropy",
            "stats/mean_Q1",
            "stats/mean_Q2",
        ]
        if tag in scalar_tags
    }
    config_snapshot = parse_logged_config(run_dir / "log.log")
    payload = {
        "name": run_dir.name,
        "path": run_dir.as_posix(),
        "kind": "live_run",
        "updated_at": datetime.fromtimestamp((run_dir / "summary.csv").stat().st_mtime, tz=timezone.utc).isoformat(),
        "episodes": int(len(frame)),
        "best_reward": float(pd.to_numeric(frame["ep_reward"], errors="coerce").max()) if "ep_reward" in frame.columns and not frame.empty else 0.0,
        "best_progress_laps": float(pd.to_numeric(frame["lap_progress_laps"], errors="coerce").max()) if "lap_progress_laps" in frame.columns and not frame.empty else 0.0,
        "completed_laps": float(pd.to_numeric(frame["completed_lap_count"], errors="coerce").max()) if "completed_lap_count" in frame.columns and not frame.empty else 0.0,
        "latest_episode": int(latest.get("ep_count", 0)) if latest else 0,
        "latest_reward": float(latest.get("ep_reward", 0.0)) if latest else 0.0,
        "latest_progress_laps": float(latest.get("lap_progress_laps", 0.0)) if latest else 0.0,
        "latest_speed_mean": float(latest.get("speed_mean", 0.0)) if latest else 0.0,
        "latest_packages_lost": float(latest.get("packages_lost", 0.0)) if latest else 0.0,
        "planner_model_name": str(latest.get("planner_model_name", "unknown")) if latest else "unknown",
        "profile_summary": profile,
        "recent_rows": frame.tail(RECENT_EPISODES).fillna("").to_dict(orient="records"),
        "series": {
            "episode": numeric_series(frame, "ep_count"),
            "reward": numeric_series(frame, "ep_reward"),
            "progress": numeric_series(frame, "lap_progress_laps"),
            "speed": numeric_series(frame, "speed_mean"),
            "loss": numeric_series(frame, "packages_lost"),
        },
        "artifacts": {
            "summary_csv": (run_dir / "summary.csv").as_posix(),
            "profile_summary": (run_dir / "profile_summary.json").as_posix() if (run_dir / "profile_summary.json").exists() else None,
            "time_budget_summary": (run_dir / "time_budget_summary.json").as_posix() if (run_dir / "time_budget_summary.json").exists() else None,
            "metrics_png": (run_dir / "live_training_metrics.png").as_posix() if (run_dir / "live_training_metrics.png").exists() else None,
            "trajectory_png": (run_dir / "live_trajectory_overlay.png").as_posix() if (run_dir / "live_trajectory_overlay.png").exists() else None,
            "log": (run_dir / "log.log").as_posix() if (run_dir / "log.log").exists() else None,
            "live_scalars": (run_dir / "live_scalars.csv").as_posix() if (run_dir / "live_scalars.csv").exists() else None,
            "episodes_stats": (run_dir / "episodes_stats.csv").as_posix() if (run_dir / "episodes_stats.csv").exists() else None,
        },
        "log_tail": tail_lines(run_dir / "log.log"),
        "scalar_tags": scalar_tags,
        "scalar_latest": scalar_latest,
        "scalar_series": scalar_series,
        "flow": summarize_training_flow(frame),
        "hyperparameters": select_hyperparameters(config_snapshot),
        "model_metrics": {
            "planner_model_name": str(latest.get("planner_model_name", "unknown")) if latest else "unknown",
            "latest_alpha": scalar_latest.get("stats/alpha"),
            "latest_entropy": scalar_latest.get("stats/entropy"),
            "latest_policy_loss": scalar_latest.get("loss/policy"),
            "latest_q_loss": scalar_latest.get("loss/Q"),
            "latest_entropy_loss": scalar_latest.get("loss/entropy"),
            "latest_q1": scalar_latest.get("stats/mean_Q1"),
            "latest_q2": scalar_latest.get("stats/mean_Q2"),
            "latest_planner_confidence": float(latest.get("planner_confidence_mean", 0.0)) if latest else 0.0,
            "latest_planner_latency_p95_ms": float(latest.get("planner_latency_p95_ms", 0.0)) if latest else 0.0,
            "latest_cache_refresh_count": float(latest.get("cache_refresh_count", 0.0)) if latest else 0.0,
            "latest_env_steps_per_sec": float(latest.get("env_steps_per_sec", 0.0)) if latest else 0.0,
            "latest_gradient_updates_per_sec": float(latest.get("gradient_updates_per_sec", 0.0)) if latest else 0.0,
            "latest_reset_time_s": float(latest.get("reset_time_s", 0.0)) if latest else 0.0,
            "latest_episode_live_time_s": float(latest.get("episode_live_time_s", 0.0)) if latest else 0.0,
            "latest_episode_wall_time_s": float(latest.get("episode_wall_time_s", 0.0)) if latest else 0.0,
        },
    }
    payload["language_preview"] = language_preview(payload)
    return payload


def summarize_pretrain_run(run_dir: Path) -> dict[str, Any]:
    metrics = json_load(run_dir / "offline_metrics.json", [])
    latest = metrics[-1] if metrics else {}
    payload = {
        "name": run_dir.name,
        "path": run_dir.as_posix(),
        "kind": "offline_pretrain",
        "updated_at": datetime.fromtimestamp((run_dir / "offline_metrics.json").stat().st_mtime, tz=timezone.utc).isoformat(),
        "epochs": int(len(metrics)),
        "latest_loss": float(latest.get("loss", 0.0)) if latest else 0.0,
        "metrics": metrics,
        "artifacts": {
            "offline_metrics": (run_dir / "offline_metrics.json").as_posix(),
            "shared_backbone_bundle": (run_dir / "shared_backbone_bundle.pth").as_posix() if (run_dir / "shared_backbone_bundle.pth").exists() else None,
            "segment_value_net": (run_dir / "segment_value_net.pth").as_posix() if (run_dir / "segment_value_net.pth").exists() else None,
        },
    }
    payload["language_preview"] = language_preview(payload)
    return payload


def summarize_segment_store(run_dir: Path) -> dict[str, Any]:
    parquet_path = run_dir / "segments.parquet"
    csv_path = run_dir / "segments.csv"
    frame = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.read_csv(csv_path)
    payload = {
        "name": run_dir.name,
        "path": run_dir.as_posix(),
        "kind": "segment_store",
        "updated_at": datetime.fromtimestamp((parquet_path if parquet_path.exists() else csv_path).stat().st_mtime, tz=timezone.utc).isoformat(),
        "segments": int(len(frame)),
        "sample": compact_segment_sample(frame),
        "artifacts": {
            "segments_parquet": parquet_path.as_posix() if parquet_path.exists() else None,
            "segments_csv": csv_path.as_posix() if csv_path.exists() else None,
        },
    }
    payload["language_preview"] = language_preview(payload)
    return payload


def summarize_sweep(run_dir: Path) -> dict[str, Any]:
    return {
        "name": run_dir.name,
        "path": run_dir.as_posix(),
        "kind": "sweep",
        "updated_at": datetime.fromtimestamp((run_dir / "status.json").stat().st_mtime, tz=timezone.utc).isoformat(),
        "status": json_load(run_dir / "status.json", {}),
    }


def classify_process(command_line: str) -> str:
    lowered = command_line.lower()
    if "dashboard.app" in lowered or "start-unifieddashboard.ps1" in lowered:
        return "dashboard"
    if "live_experiment_sweep.py" in lowered:
        return "sweep"
    if "train_for_duration.py" in lowered or " train.py" in lowered:
        return "training"
    return "other"


def extract_work_dir(command_line: str) -> str | None:
    match = re.search(r"work_dir=\s*([^\s]+)", command_line)
    return match.group(1).strip("\"'") if match else None


def list_active_processes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time", "status"]):
        try:
            cmdline = " ".join(proc.info.get("cmdline") or [])
            if not cmdline or not any(pattern in cmdline for pattern in PROCESS_PATTERNS):
                continue
            rows.append(
                {
                    "pid": int(proc.info["pid"]),
                    "name": str(proc.info.get("name") or ""),
                    "kind": classify_process(cmdline),
                    "status": str(proc.info.get("status") or ""),
                    "started_at": datetime.fromtimestamp(float(proc.info.get("create_time") or 0.0), tz=timezone.utc).isoformat(),
                    "uses_uv": (" uv " in f" {cmdline.lower()} ") or ("uv.exe" in cmdline.lower()) or ("Start-UnifiedDashboard.ps1" in cmdline),
                    "shared_backbone": "shared_sac" in cmdline or "shared_backbone_sac" in cmdline,
                    "work_dir": extract_work_dir(cmdline),
                    "command_line": cmdline,
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    rows.sort(key=lambda item: item["started_at"], reverse=True)
    return rows


def build_overview(outputs_root: Path) -> dict[str, Any]:
    live_runs: list[dict[str, Any]] = []
    offline_runs: list[dict[str, Any]] = []
    segment_stores: list[dict[str, Any]] = []
    sweeps: list[dict[str, Any]] = []
    run_dirs = sorted([p for p in outputs_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)[:MAX_RUNS]
    for run_dir in run_dirs:
        kind = detect_run_kind(run_dir)
        if kind == "live_run":
            live_runs.append(summarize_live_run(run_dir))
        elif kind == "offline_pretrain":
            offline_runs.append(summarize_pretrain_run(run_dir))
        elif kind == "segment_store":
            segment_stores.append(summarize_segment_store(run_dir))
        elif kind == "sweep":
            sweeps.append(summarize_sweep(run_dir))
    active = list_active_processes()
    active_training = [row for row in active if row["kind"] in {"training", "sweep"}]
    active_shared = [row for row in active_training if row["shared_backbone"]]
    active_legacy = [row for row in active_training if not row["shared_backbone"]]
    return {
        "generated_at": iso_now(),
        "outputs_root": outputs_root.as_posix(),
        "sim_status": {
            "management_socket_ready": probe_tcp("127.0.0.1", 2347),
            "control_socket_ready": probe_tcp("127.0.0.1", 2345),
            "dashboard_port_ready": probe_tcp("127.0.0.1", 8090),
        },
        "lan_addresses": local_ipv4_addresses(),
        "counts": {
            "live_runs": len(live_runs),
            "offline_runs": len(offline_runs),
            "segment_stores": len(segment_stores),
            "sweeps": len(sweeps),
            "active_training_processes": len(active_training),
        },
        "training_status": {
            "is_training_running": bool(active_training),
            "active_processes": active_training,
            "active_shared_backbone_processes": active_shared,
            "active_legacy_processes": active_legacy,
            "dashboard_processes": [row for row in active if row["kind"] == "dashboard"],
        },
        "architecture": ARCHITECTURE,
        "best_live": max(live_runs, key=lambda item: item["best_progress_laps"], default=None),
        "latest_offline": offline_runs[0] if offline_runs else None,
        "live_runs": live_runs,
        "offline_runs": offline_runs,
        "segment_stores": segment_stores,
        "sweeps": sweeps,
    }


HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AC Unified Backbone Dashboard</title>
<style>
body{margin:0;font-family:Georgia,serif;background:#f6f1e5;color:#1d2a2a}
.shell{max-width:1500px;margin:0 auto;padding:24px}
.panel{background:#fffdf7;border:1px solid #d7d0c2;border-radius:18px;padding:16px 18px;box-shadow:0 14px 36px rgba(0,0,0,.06)}
.hero,.split,.overview{display:grid;gap:16px}
.hero{grid-template-columns:2fr 1fr}
.overview{grid-template-columns:.9fr 1.25fr .95fr}
.split{grid-template-columns:1fr 1fr}
.tabs{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}
.tab{padding:9px 12px;border-radius:999px;border:1px solid #d4cbba;background:#f8f4ea;cursor:pointer}
.tab.active{background:#0f766e;color:#fff;border-color:#0f766e}
.tab-panel{display:none}
.tab-panel.active{display:block}
.eyebrow{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:#5e6b67;margin-bottom:8px}
.grid4{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:14px}
.metric{background:#faf7ef;border:1px solid #e8e1d0;border-radius:14px;padding:12px}
.label{font-size:11px;text-transform:uppercase;color:#5e6b67}
.value{font-size:24px;margin-top:5px}
.stack,.list{display:flex;flex-direction:column;gap:10px}
.list{max-height:72vh;overflow:auto}
.card{border:1px solid #ded6c6;border-radius:14px;padding:12px;background:#fbf8f0}
.run{cursor:pointer}
.run.active{border-color:#0f766e;box-shadow:inset 0 0 0 1px #0f766e}
.muted{color:#5e6b67}
.tiny{font-size:12px}
.pill{display:inline-flex;gap:8px;align-items:center;padding:7px 10px;border:1px solid #ded6c6;border-radius:999px;background:#f7f4ec;margin:4px 8px 0 0;font-size:12px}
.dot{width:10px;height:10px;border-radius:50%;background:#b91c1c}
.dot.good{background:#166534}
.chart{height:150px;border:1px solid #e8dfd0;border-radius:14px;background:#faf7ef;padding:8px}
.chartgrid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.chartstack{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
.mlgrid{display:grid;grid-template-columns:1.2fr .8fr;gap:16px}
.mlstack{display:flex;flex-direction:column;gap:16px}
.tablewrap{max-height:380px;overflow:auto}
svg{width:100%;height:100%}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:7px 5px;border-bottom:1px solid #eee6d8;text-align:left;vertical-align:top}
pre{white-space:pre-wrap;font-size:12px;font-family:Consolas,monospace;margin:0}
ul{margin:0;padding-left:18px}
code{font-family:Consolas,monospace}
@media (max-width:1200px){.hero,.overview,.split,.grid4,.chartgrid,.chartstack,.mlgrid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="shell">
<div class="hero">
<div class="panel">
<div class="eyebrow">Unified Backbone RL</div>
<h1>Training, architecture, testing, and grounded language details.</h1>
<p class="muted">This view distinguishes active training from completed runs and exposes architecture, phases, and language diagnostics.</p>
<div class="grid4" id="kpis"></div>
</div>
<div class="panel">
<div class="eyebrow">Runtime</div>
<div id="status"></div>
<div class="stack" style="margin-top:14px">
<div><h3>Training State</h3><div id="trainingState" class="tiny muted"></div></div>
<div><h3>LAN Access</h3><div id="lan" class="tiny muted"></div></div>
</div>
</div>
</div>
<div class="tabs">
<button class="tab active" data-tab="overview">Overview</button>
<button class="tab" data-tab="metrics">ML</button>
<button class="tab" data-tab="architecture">Architecture</button>
<button class="tab" data-tab="training">Training</button>
<button class="tab" data-tab="language">Language</button>
</div>
<section id="tab-overview" class="tab-panel active">
<div class="overview">
<div class="panel">
<div class="eyebrow">Artifacts</div>
<h3>Live Runs</h3><div id="liveRuns" class="list"></div>
<h3 style="margin-top:12px">Offline Pretrains</h3><div id="offlineRuns" class="list"></div>
<h3 style="margin-top:12px">Segment Stores</h3><div id="segmentRuns" class="list"></div>
</div>
<div class="panel">
<div class="eyebrow">Selected</div>
<div id="selectedHeader" class="stack"></div>
<div id="selectedMain" class="stack" style="margin-top:14px"></div>
</div>
<div class="panel">
<div class="eyebrow">Details</div>
<div id="selectedDetails" class="stack"></div>
</div>
</div>
</section>
<section id="tab-metrics" class="tab-panel">
<div class="mlgrid">
<div class="mlstack">
<div class="panel">
<div class="eyebrow">Optimization</div>
<div id="mlHeadline" class="grid4"></div>
<div id="optimCharts" class="chartstack" style="margin-top:14px"></div>
</div>
<div class="panel">
<div class="eyebrow">Rollout Flow</div>
<div id="flowCharts" class="chartstack"></div>
</div>
</div>
<div class="mlstack">
<div class="panel">
<div class="eyebrow">Hyperparameters</div>
<div id="hyperparams" class="stack"></div>
</div>
<div class="panel">
<div class="eyebrow">Latest Model State</div>
<div id="modelState" class="stack"></div>
</div>
<div class="panel">
<div class="eyebrow">Recent Flow Rows</div>
<div id="flowTable" class="tablewrap"></div>
</div>
</div>
</div>
</section>
<section id="tab-architecture" class="tab-panel">
<div class="split">
<div class="panel">
<div class="eyebrow">Architecture</div>
<h2 id="archTitle"></h2>
<div id="archStatus" class="stack" style="margin-top:12px"></div>
<h3 style="margin-top:12px">Components</h3>
<ul id="archComponents"></ul>
</div>
<div class="panel">
<div class="eyebrow">Phases</div>
<h3>Execution phases</h3>
<ul id="archPhases"></ul>
<h3 style="margin-top:12px">Current blockers</h3>
<ul id="archBlockers"></ul>
<h3 style="margin-top:12px">Docs</h3>
<ul id="archDocs"></ul>
</div>
</div>
</section>
<section id="tab-training" class="tab-panel">
<div class="split">
<div class="panel">
<div class="eyebrow">Active Processes</div>
<div id="activeProcesses" class="stack"></div>
</div>
<div class="panel">
<div class="eyebrow">Sweeps</div>
<div id="sweeps" class="stack"></div>
</div>
</div>
</section>
<section id="tab-language" class="tab-panel">
<div class="split">
<div class="panel">
<div class="eyebrow">Grounded Language Output</div>
<div id="languagePreview" class="stack"></div>
</div>
<div class="panel">
<div class="eyebrow">System Notes</div>
<ul>
<li>Language output shown here is grounded on telemetry and run metrics.</li>
<li>True Qwen 3.5 runtime is not active yet in this environment.</li>
<li>Current previews are diagnostic and action-oriented.</li>
</ul>
</div>
</div>
</section>
</div>
<script>
let data=null,selectedPath=null;
const esc=(v)=>String(v??"").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
const fmt=(v,d=2)=>v===null||v===undefined||Number.isNaN(Number(v))?"-":Number(v).toFixed(d);
const metric=(l,v)=>`<div class="metric"><div class="label">${l}</div><div class="value">${v}</div></div>`;
function tabs(){document.querySelectorAll(".tab").forEach((b)=>{b.onclick=()=>{document.querySelectorAll(".tab,.tab-panel").forEach((n)=>n.classList.remove("active"));b.classList.add("active");document.getElementById("tab-"+b.dataset.tab).classList.add("active");};});}
function poly(values,stroke){if(!values||values.length<2){return `<div class="tiny muted">Not enough data yet.</div>`;}const mi=Math.min(...values),ma=Math.max(...values),ra=Math.max(ma-mi,1e-6);const pts=values.map((v,i)=>`${(i/Math.max(values.length-1,1))*100},${100-(((v-mi)/ra)*92+4)}`).join(" ");return `<svg viewBox="0 0 100 100"><polyline fill="none" stroke="${stroke}" stroke-width="2.2" points="${pts}" /></svg>`;}
function multiPoly(series){const valid=(series||[]).filter((item)=>item.values&&item.values.length>1);if(valid.length===0){return `<div class="tiny muted">Not enough data yet.</div>`;}const all=valid.flatMap((item)=>item.values);const mi=Math.min(...all),ma=Math.max(...all),ra=Math.max(ma-mi,1e-6);const lines=valid.map((item)=>{const pts=item.values.map((v,i)=>`${(i/Math.max(item.values.length-1,1))*100},${100-(((v-mi)/ra)*92+4)}`).join(" ");return `<polyline fill="none" stroke="${item.stroke}" stroke-width="2.2" points="${pts}" />`;}).join("");return `<svg viewBox="0 0 100 100">${lines}</svg>`;}
function card(run,extra){const active=selectedPath===run.path?"active":"";return `<div class="card run ${active}" data-path="${run.path}"><strong>${esc(run.name)}</strong><div class="tiny muted">${extra}</div><div class="tiny muted">${new Date(run.updated_at).toLocaleString()}</div></div>`;}
function bind(){document.querySelectorAll(".run").forEach((n)=>{n.onclick=()=>{selectedPath=n.dataset.path;renderAll(data);};});}
function artifacts(obj){const rows=Object.entries(obj||{}).filter(([,v])=>v).map(([k,v])=>`<li><strong>${esc(k)}</strong><br><span class="tiny">${esc(v)}</span></li>`);return rows.length?`<ul>${rows.join("")}</ul>`:`<div class="tiny muted">No artifacts recorded.</div>`;}
function chartCard(title, body, subtitle){return `<div class="card"><strong>${esc(title)}</strong>${subtitle?`<div class="tiny muted" style="margin-top:4px">${esc(subtitle)}</div>`:""}<div class="chart" style="margin-top:10px">${body}</div></div>`;}
function kvTable(rows){if(!rows||rows.length===0){return `<div class="tiny muted">No values recorded.</div>`;}return `<table><tbody>${rows.map((row)=>`<tr><th>${esc(row[0])}</th><td>${esc(row[1])}</td></tr>`).join("")}</tbody></table>`;}
function renderSelected(run){if(!run){document.getElementById("selectedHeader").innerHTML="<h3>No run selected.</h3>";document.getElementById("selectedMain").innerHTML="";document.getElementById("selectedDetails").innerHTML="";document.getElementById("languagePreview").innerHTML="<div class='tiny muted'>Select a run.</div>";return;}
document.getElementById("selectedHeader").innerHTML=`<h2>${esc(run.name)}</h2><div class="tiny muted">${esc(run.path)}</div>`;
if(run.kind==="live_run"){const p=run.profile_summary||{};const rows=(run.recent_rows||[]).slice().reverse().slice(0,10).map((r)=>`<tr><td>${r.ep_count??""}</td><td>${fmt(r.ep_reward,1)}</td><td>${fmt(r.lap_progress_laps,3)}</td><td>${fmt(r.speed_mean,2)}</td><td>${fmt(r.packages_lost,0)}</td></tr>`).join("");document.getElementById("selectedMain").innerHTML=`<div class="grid4">${metric("Episodes",run.episodes)}${metric("Best Progress",fmt(run.best_progress_laps,3)+" laps")}${metric("Best Reward",fmt(run.best_reward,1))}${metric("Latest Speed",fmt(run.latest_speed_mean,2)+" m/s")}</div><div class="chartgrid"><div class="chart">${poly(run.series.reward,"#0f766e")}</div><div class="chart">${poly(run.series.progress,"#b45309")}</div><div class="chart">${poly(run.series.speed,"#1d4ed8")}</div><div class="chart">${poly(run.series.loss,"#b91c1c")}</div></div>`;document.getElementById("selectedDetails").innerHTML=`<div><h3>Profile</h3><table><tr><th>Env steps / sec</th><td>${fmt(p.env_steps_per_sec_wall_clock,2)}</td></tr><tr><th>Gradient updates / sec</th><td>${fmt(p.gradient_updates_per_sec_wall_clock,2)}</td></tr><tr><th>Best laps / hour</th><td>${fmt(p.best_lap_progress_laps_per_hour,3)}</td></tr><tr><th>Mean step perf</th><td>${fmt(p.mean_step_perf_s,4)} s</td></tr><tr><th>Mean packets lost</th><td>${fmt(p.mean_packages_lost_per_episode,0)}</td></tr></table></div><div><h3>Recent Episodes</h3><table><thead><tr><th>Ep</th><th>Reward</th><th>Progress</th><th>Speed</th><th>Packets</th></tr></thead><tbody>${rows}</tbody></table></div><div><h3>Artifacts</h3>${artifacts(run.artifacts)}</div><div><h3>Log Tail</h3><pre>${esc((run.log_tail||[]).join("\\n"))}</pre></div>`;}
else if(run.kind==="offline_pretrain"){const rows=(run.metrics||[]).map((r)=>`<tr><td>${r.epoch}</td><td>${fmt(r.loss,6)}</td></tr>`).join("");document.getElementById("selectedMain").innerHTML=`<div class="grid4">${metric("Epochs",run.epochs)}${metric("Latest Loss",fmt(run.latest_loss,4))}${metric("Bundle",run.artifacts.shared_backbone_bundle?"yes":"no")}${metric("Value Net",run.artifacts.segment_value_net?"yes":"no")}</div><div class="chart">${poly((run.metrics||[]).map((r)=>Number(r.loss||0)),"#7c3aed")}</div>`;document.getElementById("selectedDetails").innerHTML=`<div><h3>Offline Metrics</h3><table><thead><tr><th>Epoch</th><th>Loss</th></tr></thead><tbody>${rows}</tbody></table></div><div><h3>Artifacts</h3>${artifacts(run.artifacts)}</div>`;}
else{document.getElementById("selectedMain").innerHTML=`<div class="grid4">${metric("Segments",run.segments)}${metric("Track",run.sample.track_id||"-")}${metric("Car",run.sample.car_id||"-")}${metric("Sample",run.sample.segment_id?"yes":"no")}</div>`;document.getElementById("selectedDetails").innerHTML=`<div><h3>Artifacts</h3>${artifacts(run.artifacts)}</div><div><h3>Sample Row</h3><pre>${esc(JSON.stringify(run.sample||{},null,2))}</pre></div>`;}
document.getElementById("languagePreview").innerHTML=`<div class="card"><h3>Preview</h3><ul>${(run.language_preview||[]).map((x)=>`<li>${esc(x)}</li>`).join("")}</ul></div>`;}
function renderMetrics(run){
if(!run){document.getElementById("mlHeadline").innerHTML="";document.getElementById("optimCharts").innerHTML=`<div class="tiny muted">Select a run.</div>`;document.getElementById("flowCharts").innerHTML="";document.getElementById("hyperparams").innerHTML="";document.getElementById("modelState").innerHTML="";document.getElementById("flowTable").innerHTML="";return;}
if(run.kind==="live_run"){
const model=run.model_metrics||{}, scalars=run.scalar_series||{}, flow=run.flow||{}, hyper=run.hyperparameters||{};
document.getElementById("mlHeadline").innerHTML=[metric("Policy Loss",fmt(model.latest_policy_loss,3)),metric("Q Loss",fmt(model.latest_q_loss,3)),metric("Alpha",fmt(model.latest_alpha,4)),metric("Entropy",fmt(model.latest_entropy,3))].join("");
document.getElementById("optimCharts").innerHTML=[
chartCard("Policy Loss",poly((scalars["loss/policy"]||{}).values,"#0f766e"),`latest ${fmt(model.latest_policy_loss,4)}`),
chartCard("Q Loss",poly((scalars["loss/Q"]||{}).values,"#b45309"),`latest ${fmt(model.latest_q_loss,4)}`),
chartCard("Entropy Loss",poly((scalars["loss/entropy"]||{}).values,"#7c3aed"),`latest ${fmt(model.latest_entropy_loss,4)}`),
chartCard("Alpha / Entropy",multiPoly([{values:(scalars["stats/alpha"]||{}).values,stroke:"#1d4ed8"},{values:(scalars["stats/entropy"]||{}).values,stroke:"#b91c1c"}]),`alpha ${fmt(model.latest_alpha,4)} | entropy ${fmt(model.latest_entropy,3)}`),
chartCard("Q1 / Q2",multiPoly([{values:(scalars["stats/mean_Q1"]||{}).values,stroke:"#0f766e"},{values:(scalars["stats/mean_Q2"]||{}).values,stroke:"#7c3aed"}]),`Q1 ${fmt(model.latest_q1,3)} | Q2 ${fmt(model.latest_q2,3)}`),
].join("");
document.getElementById("flowCharts").innerHTML=[
chartCard("Reward / Progress",multiPoly([{values:run.series.reward||[],stroke:"#0f766e"},{values:run.series.progress||[],stroke:"#b45309"}]),`best progress ${fmt(run.best_progress_laps,3)} laps`),
chartCard("Throughput",multiPoly([{values:flow.env_steps_per_sec||[],stroke:"#1d4ed8"},{values:flow.gradient_updates_per_sec||[],stroke:"#b91c1c"}]),`env ${fmt(model.latest_env_steps_per_sec,2)} | grad ${fmt(model.latest_gradient_updates_per_sec,2)}`),
chartCard("Phase Timing",multiPoly([{values:flow.reset_time_s||[],stroke:"#7c3aed"},{values:flow.live_time_s||[],stroke:"#0f766e"},{values:flow.wall_time_s||[],stroke:"#b45309"}]),`reset ${fmt(model.latest_reset_time_s,2)}s | live ${fmt(model.latest_episode_live_time_s,2)}s`),
chartCard("Planner / Systems",multiPoly([{values:flow.planner_confidence_mean||[],stroke:"#166534"},{values:flow.planner_latency_p95_ms||[],stroke:"#dc2626"},{values:run.series.loss||[],stroke:"#92400e"}]),`confidence ${fmt(model.latest_planner_confidence,3)} | latency ${fmt(model.latest_planner_latency_p95_ms,2)} ms`),
].join("");
document.getElementById("hyperparams").innerHTML=Object.entries(hyper).map(([section,values])=>`<div class="card"><strong>${esc(section)}</strong>${kvTable(Object.entries(values).map(([k,v])=>[k,Array.isArray(v)?v.join(", "):String(v)]))}</div>`).join("")||`<div class="tiny muted">No hyperparameter snapshot found.</div>`;
document.getElementById("modelState").innerHTML=kvTable([
["planner_model_name", model.planner_model_name ?? "-"],
["planner_confidence_mean", fmt(model.latest_planner_confidence,4)],
["planner_latency_p95_ms", fmt(model.latest_planner_latency_p95_ms,2)],
["cache_refresh_count", fmt(model.latest_cache_refresh_count,0)],
["env_steps_per_sec", fmt(model.latest_env_steps_per_sec,2)],
["gradient_updates_per_sec", fmt(model.latest_gradient_updates_per_sec,2)],
["scalar_tags", (run.scalar_tags||[]).join(", ") || "-"],
]);
const flowRows=(flow.recent_rows||[]).slice().reverse().slice(0,10);
document.getElementById("flowTable").innerHTML=flowRows.length?`<table><thead><tr><th>Ep</th><th>Reset s</th><th>Live s</th><th>Wall s</th><th>Env/s</th><th>Grad/s</th><th>Conf</th><th>Latency</th></tr></thead><tbody>${flowRows.map((r)=>`<tr><td>${r.ep_count??""}</td><td>${fmt(r.reset_time_s,2)}</td><td>${fmt(r.episode_live_time_s,2)}</td><td>${fmt(r.episode_wall_time_s,2)}</td><td>${fmt(r.env_steps_per_sec,2)}</td><td>${fmt(r.gradient_updates_per_sec,2)}</td><td>${fmt(r.planner_confidence_mean,4)}</td><td>${fmt(r.planner_latency_p95_ms,2)}</td></tr>`).join("")}</tbody></table>`:`<div class="tiny muted">No flow rows yet.</div>`;
return;}
if(run.kind==="offline_pretrain"){document.getElementById("mlHeadline").innerHTML=[metric("Epochs",run.epochs),metric("Latest Loss",fmt(run.latest_loss,4)),metric("Bundle",run.artifacts.shared_backbone_bundle?"yes":"no"),metric("Value Net",run.artifacts.segment_value_net?"yes":"no")].join("");document.getElementById("optimCharts").innerHTML=chartCard("Offline Loss",poly((run.metrics||[]).map((r)=>Number(r.loss||0)),"#7c3aed"),`latest ${fmt(run.latest_loss,4)}`);document.getElementById("flowCharts").innerHTML=`<div class="tiny muted">Offline pretraining has no rollout-flow view.</div>`;document.getElementById("hyperparams").innerHTML=`<div class="tiny muted">Hyperparameter snapshot is not stored for this offline artifact yet.</div>`;document.getElementById("modelState").innerHTML=kvTable([["epochs", String(run.epochs)],["latest_loss", fmt(run.latest_loss,6)],["shared_backbone_bundle", run.artifacts.shared_backbone_bundle?"present":"missing"],["segment_value_net", run.artifacts.segment_value_net?"present":"missing"]]);document.getElementById("flowTable").innerHTML=`<div class="tiny muted">No episode flow for offline pretraining.</div>`;return;}
document.getElementById("mlHeadline").innerHTML=[metric("Segments",run.segments),metric("Track",run.sample.track_id||"-"),metric("Car",run.sample.car_id||"-"),metric("Sample",run.sample.segment_id?"yes":"no")].join("");document.getElementById("optimCharts").innerHTML=`<div class="tiny muted">Segment stores do not contain live optimizer curves.</div>`;document.getElementById("flowCharts").innerHTML=`<div class="tiny muted">Segment stores do not contain rollout flow.</div>`;document.getElementById("hyperparams").innerHTML=`<div class="tiny muted">No live hyperparameter snapshot for segment stores.</div>`;document.getElementById("modelState").innerHTML=kvTable(Object.entries(run.sample||{}).map(([k,v])=>[k,String(v??"-")]));document.getElementById("flowTable").innerHTML=`<div class="tiny muted">No flow rows for segment stores.</div>`;}
function renderAll(payload){data=payload;if(!selectedPath){selectedPath=payload.live_runs[0]?.path||payload.offline_runs[0]?.path||payload.segment_stores[0]?.path||null;}
document.getElementById("kpis").innerHTML=[metric("Active Training",payload.counts.active_training_processes),metric("Best Progress",fmt((payload.best_live||{}).best_progress_laps||0,3)+" laps"),metric("Offline Loss",payload.latest_offline?fmt(payload.latest_offline.latest_loss,3):"-"),metric("Segment Stores",payload.counts.segment_stores)].join("");
const s=payload.sim_status||{};document.getElementById("status").innerHTML=`<div class="pill"><span class="dot ${s.management_socket_ready?"good":""}"></span>Mgmt 2347 ${s.management_socket_ready?"ready":"down"}</div><div class="pill"><span class="dot ${s.control_socket_ready?"good":""}"></span>Ctrl 2345 ${s.control_socket_ready?"ready":"down"}</div><div class="pill"><span class="dot ${s.dashboard_port_ready?"good":""}"></span>Dashboard 8090 ${s.dashboard_port_ready?"ready":"down"}</div>`;
const t=payload.training_status||{};let trainingLine="No training process is active right now. The latest shared-backbone test run has finished.";if(t.is_training_running){const shared=(t.active_shared_backbone_processes||[]).length;const legacy=(t.active_legacy_processes||[]).length;trainingLine=`Training is running now. ${t.active_processes.length} active training or sweep process(es) detected. Shared-backbone: ${shared}. Legacy SAC/stage5: ${legacy}.`;}document.getElementById("trainingState").innerHTML=trainingLine;
document.getElementById("lan").innerHTML=(payload.lan_addresses||[]).map((ip)=>`<div>http://${ip}:8090</div>`).join("");
document.getElementById("liveRuns").innerHTML=payload.live_runs.map((r)=>card(r,`episodes ${r.episodes} | best ${fmt(r.best_progress_laps,3)} laps | speed ${fmt(r.latest_speed_mean,2)} m/s`)).join("")||`<div class="tiny muted">No live runs.</div>`;
document.getElementById("offlineRuns").innerHTML=payload.offline_runs.map((r)=>card(r,`epochs ${r.epochs} | loss ${fmt(r.latest_loss,4)}`)).join("")||`<div class="tiny muted">No offline runs.</div>`;
document.getElementById("segmentRuns").innerHTML=payload.segment_stores.map((r)=>card(r,`segments ${r.segments}`)).join("")||`<div class="tiny muted">No segment stores.</div>`;
bind();
const arch=payload.architecture||{};document.getElementById("archTitle").textContent=arch.title||"";document.getElementById("archStatus").innerHTML=Object.entries(arch.status||{}).map(([k,v])=>`<div class="card"><strong>${esc(k.replaceAll("_"," "))}</strong><br>${esc(v)}</div>`).join("");document.getElementById("archComponents").innerHTML=(arch.components||[]).map((x)=>`<li>${esc(x)}</li>`).join("");document.getElementById("archPhases").innerHTML=(arch.phases||[]).map((x)=>`<li>${esc(x)}</li>`).join("");document.getElementById("archBlockers").innerHTML=(arch.blockers||[]).map((x)=>`<li>${esc(x)}</li>`).join("");document.getElementById("archDocs").innerHTML=(arch.docs||[]).map((x)=>`<li>${esc(x)}</li>`).join("");
document.getElementById("activeProcesses").innerHTML=(t.active_processes||[]).map((p)=>`<div class="card"><strong>${esc(p.kind)}</strong> PID ${p.pid}<br><span class="tiny muted">uv: ${p.uses_uv?"yes":"no"} | shared backbone: ${p.shared_backbone?"yes":"no"}</span>${p.work_dir?`<br><span class="tiny muted">work_dir: ${esc(p.work_dir)}</span>`:""}<pre>${esc(p.command_line)}</pre></div>`).join("")||`<div class="tiny muted">No active training process detected.</div>`;
document.getElementById("sweeps").innerHTML=(payload.sweeps||[]).slice(0,6).map((sw)=>`<div class="card"><strong>${esc(sw.name)}</strong><pre>${esc(JSON.stringify(sw.status||{},null,2))}</pre></div>`).join("")||`<div class="tiny muted">No sweeps found.</div>`;
const selected=[...payload.live_runs,...payload.offline_runs,...payload.segment_stores].find((r)=>r.path===selectedPath);renderSelected(selected);renderMetrics(selected);}
async function refresh(){const res=await fetch("/api/overview");renderAll(await res.json());}
tabs();refresh();setInterval(refresh,10000);
</script>
</body>
</html>
"""


def build_app(outputs_root: Path) -> FastAPI:
    app = FastAPI(title="AC Unified Backbone Dashboard", version="0.2.0")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return HTML

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        return sanitize_for_json(build_overview(outputs_root))

    @app.get("/api/run")
    def run_details(path: str) -> dict[str, Any]:
        run_dir = Path(path).resolve()
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail="Run path not found.")
        if outputs_root.resolve() not in run_dir.parents and run_dir != outputs_root.resolve():
            raise HTTPException(status_code=400, detail="Run path is outside outputs root.")
        kind = detect_run_kind(run_dir)
        if kind == "live_run":
            return sanitize_for_json(summarize_live_run(run_dir))
        if kind == "offline_pretrain":
            return sanitize_for_json(summarize_pretrain_run(run_dir))
        if kind == "segment_store":
            return sanitize_for_json(summarize_segment_store(run_dir))
        if kind == "sweep":
            return sanitize_for_json(summarize_sweep(run_dir))
        raise HTTPException(status_code=404, detail="Unsupported run type.")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Assetto Corsa unified-backbone dashboard.")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs_root = Path(args.outputs_root).resolve()
    outputs_root.mkdir(parents=True, exist_ok=True)
    uvicorn.run(build_app(outputs_root), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
