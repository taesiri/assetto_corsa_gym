import argparse
import json
import logging
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    overrides: tuple[str, ...]


STAGE1_EXPERIMENTS = (
    Experiment(
        name="baseline_bias",
        description="Current throttle-forward prior without overspeed shaping.",
        overrides=(),
    ),
    Experiment(
        name="safer_braking_prior",
        description="Reduce throttle push and add more braking authority before corners.",
        overrides=(
            "AssettoCorsa.action_bias_steer_base=0.74",
            "AssettoCorsa.action_bias_throttle_delta=0.08",
            "AssettoCorsa.action_bias_brake_delta=0.30",
            "AssettoCorsa.action_bias_brake_release=0.10",
        ),
    ),
    Experiment(
        name="overspeed_mild",
        description="Mild curvature-speed penalty plus overspeed brake assist.",
        overrides=(
            "AssettoCorsa.use_heuristic_speed_reward=True",
            "AssettoCorsa.heuristic_target_speed_min=14.0",
            "AssettoCorsa.heuristic_target_speed_max=46.0",
            "AssettoCorsa.heuristic_target_speed_curvature_scale=0.018",
            "AssettoCorsa.heuristic_target_speed_power=0.70",
            "AssettoCorsa.overspeed_penalty_coef=0.22",
            "AssettoCorsa.action_bias_throttle_delta=0.10",
            "AssettoCorsa.action_bias_brake_delta=0.28",
            "AssettoCorsa.action_bias_brake_release=0.12",
            "AssettoCorsa.action_bias_overspeed_throttle_cut=0.20",
            "AssettoCorsa.action_bias_overspeed_brake_gain=0.25",
        ),
    ),
    Experiment(
        name="overspeed_aggressive",
        description="Stronger target-speed penalty and harder corner-entry braking.",
        overrides=(
            "AssettoCorsa.use_heuristic_speed_reward=True",
            "AssettoCorsa.heuristic_target_speed_min=12.0",
            "AssettoCorsa.heuristic_target_speed_max=44.0",
            "AssettoCorsa.heuristic_target_speed_curvature_scale=0.016",
            "AssettoCorsa.heuristic_target_speed_power=0.82",
            "AssettoCorsa.overspeed_penalty_coef=0.35",
            "AssettoCorsa.action_bias_steer_base=0.78",
            "AssettoCorsa.action_bias_throttle_delta=0.06",
            "AssettoCorsa.action_bias_brake_delta=0.34",
            "AssettoCorsa.action_bias_brake_release=0.06",
            "AssettoCorsa.action_bias_overspeed_throttle_cut=0.28",
            "AssettoCorsa.action_bias_overspeed_brake_gain=0.38",
        ),
    ),
    Experiment(
        name="corner_release",
        description="Keep more steering authority while using overspeed-aware braking.",
        overrides=(
            "AssettoCorsa.use_heuristic_speed_reward=True",
            "AssettoCorsa.heuristic_target_speed_min=13.0",
            "AssettoCorsa.heuristic_target_speed_max=45.0",
            "AssettoCorsa.heuristic_target_speed_curvature_scale=0.017",
            "AssettoCorsa.heuristic_target_speed_power=0.76",
            "AssettoCorsa.overspeed_penalty_coef=0.28",
            "AssettoCorsa.action_bias_steer_base=0.55",
            "AssettoCorsa.action_bias_throttle_delta=0.09",
            "AssettoCorsa.action_bias_brake_delta=0.30",
            "AssettoCorsa.action_bias_brake_release=0.10",
            "AssettoCorsa.action_bias_overspeed_throttle_cut=0.22",
            "AssettoCorsa.action_bias_overspeed_brake_gain=0.30",
        ),
    ),
)

STAGE2_EXPERIMENTS = (
    Experiment(
        name="winner_baseline",
        description="Stage-2 control using the winning mild overspeed shaping.",
        overrides=(
            "AssettoCorsa.use_heuristic_speed_reward=True",
            "AssettoCorsa.heuristic_target_speed_min=14.0",
            "AssettoCorsa.heuristic_target_speed_max=46.0",
            "AssettoCorsa.heuristic_target_speed_curvature_scale=0.018",
            "AssettoCorsa.heuristic_target_speed_power=0.70",
            "AssettoCorsa.overspeed_penalty_coef=0.22",
            "AssettoCorsa.action_bias_throttle_delta=0.10",
            "AssettoCorsa.action_bias_brake_delta=0.28",
            "AssettoCorsa.action_bias_brake_release=0.12",
            "AssettoCorsa.action_bias_overspeed_throttle_cut=0.20",
            "AssettoCorsa.action_bias_overspeed_brake_gain=0.25",
        ),
    ),
    Experiment(
        name="winner_low_lr",
        description="Same winner priors with slower policy/Q updates to reduce drift.",
        overrides=(
            "SAC.policy_lr=0.00015",
            "SAC.q_lr=0.00015",
            "SAC.entropy_lr=0.0001",
            "AssettoCorsa.use_heuristic_speed_reward=True",
            "AssettoCorsa.heuristic_target_speed_min=14.0",
            "AssettoCorsa.heuristic_target_speed_max=46.0",
            "AssettoCorsa.heuristic_target_speed_curvature_scale=0.018",
            "AssettoCorsa.heuristic_target_speed_power=0.70",
            "AssettoCorsa.overspeed_penalty_coef=0.22",
            "AssettoCorsa.action_bias_throttle_delta=0.10",
            "AssettoCorsa.action_bias_brake_delta=0.28",
            "AssettoCorsa.action_bias_brake_release=0.12",
            "AssettoCorsa.action_bias_overspeed_throttle_cut=0.20",
            "AssettoCorsa.action_bias_overspeed_brake_gain=0.25",
        ),
    ),
    Experiment(
        name="winner_tighter_corner",
        description="Earlier throttle cut and stronger braking around high curvature.",
        overrides=(
            "AssettoCorsa.use_heuristic_speed_reward=True",
            "AssettoCorsa.heuristic_target_speed_min=13.5",
            "AssettoCorsa.heuristic_target_speed_max=45.0",
            "AssettoCorsa.heuristic_target_speed_curvature_scale=0.0175",
            "AssettoCorsa.heuristic_target_speed_power=0.74",
            "AssettoCorsa.overspeed_penalty_coef=0.26",
            "AssettoCorsa.action_bias_throttle_delta=0.09",
            "AssettoCorsa.action_bias_brake_delta=0.30",
            "AssettoCorsa.action_bias_brake_release=0.10",
            "AssettoCorsa.action_bias_overspeed_throttle_cut=0.24",
            "AssettoCorsa.action_bias_overspeed_brake_gain=0.30",
        ),
    ),
    Experiment(
        name="winner_steer_freer",
        description="Preserve more steering authority while keeping mild overspeed shaping.",
        overrides=(
            "AssettoCorsa.use_heuristic_speed_reward=True",
            "AssettoCorsa.heuristic_target_speed_min=14.0",
            "AssettoCorsa.heuristic_target_speed_max=46.0",
            "AssettoCorsa.heuristic_target_speed_curvature_scale=0.018",
            "AssettoCorsa.heuristic_target_speed_power=0.70",
            "AssettoCorsa.overspeed_penalty_coef=0.22",
            "AssettoCorsa.action_bias_steer_base=0.55",
            "AssettoCorsa.action_bias_throttle_delta=0.10",
            "AssettoCorsa.action_bias_brake_delta=0.29",
            "AssettoCorsa.action_bias_brake_release=0.12",
            "AssettoCorsa.action_bias_overspeed_throttle_cut=0.21",
            "AssettoCorsa.action_bias_overspeed_brake_gain=0.27",
        ),
    ),
)

STAGE3_EXPERIMENTS = (
    Experiment(
        name="winner_control_anchor",
        description="Control anchor: the stage-2 winner without any extra turn-control priors.",
        overrides=(
            "AssettoCorsa.use_heuristic_speed_reward=True",
            "AssettoCorsa.heuristic_target_speed_min=13.5",
            "AssettoCorsa.heuristic_target_speed_max=45.0",
            "AssettoCorsa.heuristic_target_speed_curvature_scale=0.0175",
            "AssettoCorsa.heuristic_target_speed_power=0.74",
            "AssettoCorsa.overspeed_penalty_coef=0.26",
            "AssettoCorsa.action_bias_throttle_delta=0.09",
            "AssettoCorsa.action_bias_brake_delta=0.30",
            "AssettoCorsa.action_bias_brake_release=0.10",
            "AssettoCorsa.action_bias_overspeed_throttle_cut=0.24",
            "AssettoCorsa.action_bias_overspeed_brake_gain=0.30",
        ),
    ),
    Experiment(
        name="turn_control_soft",
        description="Light reference steering and soft turn-speed shaping around the stable winner.",
        overrides=(
            "AssettoCorsa.use_heuristic_speed_reward=True",
            "AssettoCorsa.heuristic_target_speed_min=13.5",
            "AssettoCorsa.heuristic_target_speed_max=45.0",
            "AssettoCorsa.heuristic_target_speed_curvature_scale=0.0175",
            "AssettoCorsa.heuristic_target_speed_power=0.74",
            "AssettoCorsa.overspeed_penalty_coef=0.28",
            "AssettoCorsa.overspeed_penalty_power=1.10",
            "AssettoCorsa.turn_gap_penalty_coef=0.03",
            "AssettoCorsa.heading_error_penalty_coef=0.02",
            "AssettoCorsa.steer_change_penalty_coef=0.008",
            "AssettoCorsa.action_bias_throttle_delta=0.09",
            "AssettoCorsa.action_bias_brake_delta=0.30",
            "AssettoCorsa.action_bias_brake_release=0.10",
            "AssettoCorsa.action_bias_overspeed_throttle_cut=0.24",
            "AssettoCorsa.action_bias_overspeed_brake_gain=0.30",
            "AssettoCorsa.use_reference_steer_bias=True",
            "AssettoCorsa.reference_steer_lookahead=18.0",
            "AssettoCorsa.reference_steer_gain=0.55",
            "AssettoCorsa.reference_steer_yaw_rate_gain=0.10",
            "AssettoCorsa.reference_steer_lateral_gain=0.015",
            "AssettoCorsa.reference_steer_blend=0.28",
            "AssettoCorsa.reference_steer_speed_min=15.0",
            "AssettoCorsa.use_turn_speed_control_bias=True",
            "AssettoCorsa.turn_throttle_cap_gain=0.24",
            "AssettoCorsa.turn_brake_floor_gain=0.16",
        ),
    ),
    Experiment(
        name="turn_control_soft_low_lr",
        description="Soft turn-control stack with slower updates to reduce drift while preserving speed.",
        overrides=(
            "SAC.policy_lr=0.00015",
            "SAC.q_lr=0.00015",
            "SAC.entropy_lr=0.00010",
            "AssettoCorsa.use_heuristic_speed_reward=True",
            "AssettoCorsa.heuristic_target_speed_min=13.5",
            "AssettoCorsa.heuristic_target_speed_max=45.0",
            "AssettoCorsa.heuristic_target_speed_curvature_scale=0.0175",
            "AssettoCorsa.heuristic_target_speed_power=0.74",
            "AssettoCorsa.overspeed_penalty_coef=0.28",
            "AssettoCorsa.overspeed_penalty_power=1.10",
            "AssettoCorsa.turn_gap_penalty_coef=0.03",
            "AssettoCorsa.heading_error_penalty_coef=0.02",
            "AssettoCorsa.steer_change_penalty_coef=0.008",
            "AssettoCorsa.action_bias_throttle_delta=0.09",
            "AssettoCorsa.action_bias_brake_delta=0.30",
            "AssettoCorsa.action_bias_brake_release=0.10",
            "AssettoCorsa.action_bias_overspeed_throttle_cut=0.24",
            "AssettoCorsa.action_bias_overspeed_brake_gain=0.30",
            "AssettoCorsa.use_reference_steer_bias=True",
            "AssettoCorsa.reference_steer_lookahead=18.0",
            "AssettoCorsa.reference_steer_gain=0.55",
            "AssettoCorsa.reference_steer_yaw_rate_gain=0.10",
            "AssettoCorsa.reference_steer_lateral_gain=0.015",
            "AssettoCorsa.reference_steer_blend=0.28",
            "AssettoCorsa.reference_steer_speed_min=15.0",
            "AssettoCorsa.use_turn_speed_control_bias=True",
            "AssettoCorsa.turn_throttle_cap_gain=0.24",
            "AssettoCorsa.turn_brake_floor_gain=0.16",
        ),
    ),
    Experiment(
        name="turn_control_soft_stable_steer",
        description="Soft turn-control with slightly more steering damping but still limited intervention.",
        overrides=(
            "AssettoCorsa.use_heuristic_speed_reward=True",
            "AssettoCorsa.heuristic_target_speed_min=13.5",
            "AssettoCorsa.heuristic_target_speed_max=45.0",
            "AssettoCorsa.heuristic_target_speed_curvature_scale=0.0175",
            "AssettoCorsa.heuristic_target_speed_power=0.74",
            "AssettoCorsa.overspeed_penalty_coef=0.30",
            "AssettoCorsa.overspeed_penalty_power=1.12",
            "AssettoCorsa.turn_gap_penalty_coef=0.04",
            "AssettoCorsa.heading_error_penalty_coef=0.03",
            "AssettoCorsa.steer_change_penalty_coef=0.014",
            "AssettoCorsa.action_bias_throttle_delta=0.09",
            "AssettoCorsa.action_bias_brake_delta=0.30",
            "AssettoCorsa.action_bias_brake_release=0.10",
            "AssettoCorsa.action_bias_overspeed_throttle_cut=0.25",
            "AssettoCorsa.action_bias_overspeed_brake_gain=0.31",
            "AssettoCorsa.use_reference_steer_bias=True",
            "AssettoCorsa.reference_steer_lookahead=17.0",
            "AssettoCorsa.reference_steer_gain=0.58",
            "AssettoCorsa.reference_steer_yaw_rate_gain=0.12",
            "AssettoCorsa.reference_steer_lateral_gain=0.018",
            "AssettoCorsa.reference_steer_blend=0.34",
            "AssettoCorsa.reference_steer_speed_min=15.0",
            "AssettoCorsa.use_turn_speed_control_bias=True",
            "AssettoCorsa.turn_throttle_cap_gain=0.26",
            "AssettoCorsa.turn_brake_floor_gain=0.18",
        ),
    ),
    Experiment(
        name="turn_control_soft_brake",
        description="Soft turn-control with modestly earlier brake support for sharper entries.",
        overrides=(
            "AssettoCorsa.use_heuristic_speed_reward=True",
            "AssettoCorsa.heuristic_target_speed_min=13.25",
            "AssettoCorsa.heuristic_target_speed_max=44.5",
            "AssettoCorsa.heuristic_target_speed_curvature_scale=0.0172",
            "AssettoCorsa.heuristic_target_speed_power=0.76",
            "AssettoCorsa.overspeed_penalty_coef=0.31",
            "AssettoCorsa.overspeed_penalty_power=1.18",
            "AssettoCorsa.turn_gap_penalty_coef=0.05",
            "AssettoCorsa.heading_error_penalty_coef=0.03",
            "AssettoCorsa.steer_change_penalty_coef=0.010",
            "AssettoCorsa.action_bias_throttle_delta=0.085",
            "AssettoCorsa.action_bias_brake_delta=0.32",
            "AssettoCorsa.action_bias_brake_release=0.08",
            "AssettoCorsa.action_bias_overspeed_throttle_cut=0.28",
            "AssettoCorsa.action_bias_overspeed_brake_gain=0.34",
            "AssettoCorsa.use_reference_steer_bias=True",
            "AssettoCorsa.reference_steer_lookahead=19.0",
            "AssettoCorsa.reference_steer_gain=0.62",
            "AssettoCorsa.reference_steer_yaw_rate_gain=0.12",
            "AssettoCorsa.reference_steer_lateral_gain=0.020",
            "AssettoCorsa.reference_steer_blend=0.32",
            "AssettoCorsa.reference_steer_speed_min=14.0",
            "AssettoCorsa.use_turn_speed_control_bias=True",
            "AssettoCorsa.turn_throttle_cap_gain=0.30",
            "AssettoCorsa.turn_brake_floor_gain=0.22",
        ),
    ),
)


def get_experiments(preset: str):
    if preset == "stage3":
        return STAGE3_EXPERIMENTS
    if preset == "stage2":
        return STAGE2_EXPERIMENTS
    return STAGE1_EXPERIMENTS


def parse_args():
    parser = argparse.ArgumentParser(description="Run a live 3-hour SAC experiment sweep on AssettoCorsaGym.")
    parser.add_argument("--config", default="config.yml", help="Base config file.")
    parser.add_argument("--algo", default="sac", choices=("sac", "discor"))
    parser.add_argument("--preset", default="stage1", choices=("stage1", "stage2", "stage3"))
    parser.add_argument("--base-run-dir", required=True, help="Completed run directory that contains model/final and model/<checkpoint>.")
    parser.add_argument(
        "--base-checkpoint",
        default="best_reward",
        choices=("final", "best_reward", "best_lap_time"),
        help="Checkpoint under model/ to branch all experiments from.",
    )
    parser.add_argument(
        "--buffer-run-dir",
        default=None,
        help="Run directory that provides model/final/replay_buffer.pkl. Defaults to --base-run-dir.",
    )
    parser.add_argument(
        "--seed-run-dir",
        action="append",
        default=[],
        help="Optional run directories whose laps/ folders are loaded into the replay buffer for every experiment.",
    )
    parser.add_argument("--total-hours", type=float, default=3.0, help="Wall-clock budget for the whole sweep.")
    parser.add_argument("--probe-minutes", type=float, default=20.0, help="Per-experiment probe duration.")
    parser.add_argument("--score-episodes", type=int, default=8, help="Episodes used for rolling experiment scoring.")
    parser.add_argument("--track", default="monza")
    parser.add_argument("--car", default="ks_mazda_miata")
    parser.add_argument("--memory-size", type=int, default=300000)
    parser.add_argument("--checkpoint-freq", type=int, default=25000)
    parser.add_argument("--retry-on-failure", type=int, default=1)
    parser.add_argument(
        "--champion-checkpoint",
        default="best_reward",
        choices=("final", "best_reward", "best_lap_time"),
        help="Checkpoint used when continuing the winning experiment.",
    )
    parser.add_argument(
        "--champion-buffer-mode",
        default="stable_base",
        choices=("stable_base", "winner_final"),
        help="Replay-buffer source used for the continuation phase.",
    )
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Extra Hydra-style key=value overrides for every experiment.")
    return parser.parse_args()


def ensure_exists(path: Path, message: str):
    if not path.exists():
        raise FileNotFoundError(message)


def make_output_root(gym_root: Path) -> Path:
    output_root = gym_root / "outputs" / f"{datetime.now().strftime('%Y%m%d_%H%M%S.%f')[:-3]}_experiment_sweep"
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def numeric_mean(frame: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if column not in frame.columns:
        return default
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return default
    return float(values.mean())


def numeric_max(frame: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if column not in frame.columns:
        return default
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return default
    return float(values.max())


def positive_min(frame: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if column not in frame.columns:
        return default
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values[values > 0]
    if values.empty:
        return default
    return float(values.min())


def score_summary(summary_df: pd.DataFrame, score_episodes: int) -> dict:
    tail = summary_df.tail(max(1, min(score_episodes, len(summary_df)))).copy()
    last_reward_mean = numeric_mean(tail, "ep_reward")
    last_reward_median = float(pd.to_numeric(tail.get("ep_reward"), errors="coerce").median()) if "ep_reward" in tail else 0.0
    last_steps_mean = numeric_mean(tail, "ep_steps")
    last_speed_mean = numeric_mean(tail, "speed_mean")
    last_gap_abs_mean = numeric_mean(tail, "gap_abs_mean")
    last_overspeed_mean = numeric_mean(tail, "overspeed_mean")
    last_out_of_track_frac = numeric_mean(tail, "out_of_track_frac")
    last_heading_error_mean = numeric_mean(tail, "heading_error_mean")
    best_lap_time = positive_min(summary_df, "ep_bestLapTime")
    best_reward = numeric_max(summary_df, "ep_reward")

    score = (
        (4.0 * last_speed_mean)
        + (0.15 * last_steps_mean)
        + (0.01 * last_reward_mean)
        + (0.005 * last_reward_median)
        - (8.0 * last_gap_abs_mean)
        - (12.0 * last_overspeed_mean)
        - (60.0 * last_out_of_track_frac)
        - (30.0 * last_heading_error_mean)
    )
    if best_lap_time > 0.0:
        score += max(0.0, 500.0 - best_lap_time)

    return {
        "episodes": int(len(summary_df)),
        "total_steps": int(pd.to_numeric(summary_df["total_steps"], errors="coerce").max()) if "total_steps" in summary_df else 0,
        "best_reward": best_reward,
        "best_lap_time": best_lap_time,
        "last_reward_mean": last_reward_mean,
        "last_reward_median": last_reward_median,
        "last_speed_mean": last_speed_mean,
        "last_steps_mean": last_steps_mean,
        "last_gap_abs_mean": last_gap_abs_mean,
        "last_overspeed_mean": last_overspeed_mean,
        "last_out_of_track_frac": last_out_of_track_frac,
        "last_heading_error_mean": last_heading_error_mean,
        "score": float(score),
    }


def write_status(output_root: Path, payload: dict):
    status_path = output_root / "status.json"
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_leaderboard(output_root: Path, rows: list[dict]):
    if not rows:
        return
    leaderboard = pd.DataFrame(rows).sort_values("score", ascending=False)
    leaderboard.to_csv(output_root / "leaderboard.csv", index=False)
    (output_root / "leaderboard.json").write_text(
        leaderboard.to_json(orient="records", indent=2),
        encoding="utf-8",
    )


def launch_game(workspace_root: Path) -> bool:
    run_game = workspace_root / "scripts" / "Run-Game.ps1"
    if not run_game.exists():
        logger.warning("Cannot relaunch the game. Missing %s", run_game)
        return False
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(run_game),
            "-SkipWatcher",
        ],
        cwd=workspace_root,
        check=False,
    )
    return result.returncode == 0


def run_training(
    gym_root: Path,
    workspace_root: Path,
    experiment_name: str,
    work_dir: Path,
    load_path: Path,
    buffer_path: Path,
    seed_run_dirs: list[Path],
    duration_hours: float,
    args,
    overrides: tuple[str, ...],
):
    cmd = [
        sys.executable,
        "train_for_duration.py",
        "--config",
        args.config,
        "--algo",
        args.algo,
        "--load_path",
        str(load_path),
        "--load-buffer",
        "--buffer-path",
        str(buffer_path),
        "--duration-hours",
        str(duration_hours),
    ]

    for seed_run_dir in seed_run_dirs:
        laps_dir = seed_run_dir / "laps"
        if laps_dir.exists():
            cmd.extend(["--seed-laps-dir", str(laps_dir)])

    cmd.extend(
        [
            f"work_dir={work_dir.as_posix()}",
            "disable_wandb=True",
            "Agent.start_steps=0",
            f"Agent.memory_size={args.memory_size}",
            f"Agent.checkpoint_freq={args.checkpoint_freq}",
            f"AssettoCorsa.track={args.track}",
            f"AssettoCorsa.car={args.car}",
        ]
    )
    cmd.extend(overrides)
    cmd.extend(args.overrides)

    for attempt in range(args.retry_on_failure + 1):
        logger.info("Starting %s attempt %d: %s", experiment_name, attempt + 1, " ".join(cmd))
        result = subprocess.run(cmd, cwd=gym_root, check=False)
        if result.returncode == 0:
            return
        logger.warning("%s failed with exit code %d", experiment_name, result.returncode)
        if attempt < args.retry_on_failure and launch_game(workspace_root):
            logger.info("Relaunched the game. Retrying %s.", experiment_name)
            continue
        raise RuntimeError(f"{experiment_name} failed with exit code {result.returncode}")


def plot_run(gym_root: Path, run_dir: Path, config_path: str):
    result = subprocess.run(
        [
            sys.executable,
            "plot_live_run.py",
            "--run-dir",
            str(run_dir),
            "--config",
            config_path,
        ],
        cwd=gym_root,
        check=False,
    )
    if result.returncode != 0:
        logger.warning("Plot generation failed for %s", run_dir)


def main():
    args = parse_args()
    gym_root = Path(__file__).resolve().parent
    workspace_root = gym_root.parent
    output_root = make_output_root(gym_root)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(output_root / "orchestrator.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    base_run_dir = Path(args.base_run_dir).resolve()
    buffer_run_dir = Path(args.buffer_run_dir).resolve() if args.buffer_run_dir else base_run_dir
    seed_run_dirs = [Path(path).resolve() for path in args.seed_run_dir]
    load_path = base_run_dir / "model" / args.base_checkpoint
    buffer_path = buffer_run_dir / "model" / "final"

    ensure_exists(load_path / "policy_net.pth", f"Missing model checkpoint under {load_path}")
    ensure_exists(buffer_path / "replay_buffer.pkl", f"Missing replay buffer under {buffer_path}")

    deadline = time.time() + (args.total_hours * 3600.0)
    probe_hours = args.probe_minutes / 60.0
    experiments = get_experiments(args.preset)
    max_probe_count = min(len(experiments), max(1, math.floor((args.total_hours * 60.0) / args.probe_minutes) - 1))
    probe_experiments = list(experiments[:max_probe_count])

    run_plan = [
        {
            "name": exp.name,
            "description": exp.description,
            "overrides": list(exp.overrides),
        }
        for exp in probe_experiments
    ]
    (output_root / "experiment_plan.json").write_text(json.dumps(run_plan, indent=2), encoding="utf-8")

    leaderboard_rows = []
    write_status(
        output_root,
        {
            "stage": "probing",
            "started_at": datetime.now().isoformat(),
            "deadline": datetime.fromtimestamp(deadline).isoformat(),
            "base_run_dir": base_run_dir.as_posix(),
            "base_checkpoint": args.base_checkpoint,
            "buffer_run_dir": buffer_run_dir.as_posix(),
            "seed_run_dirs": [path.as_posix() for path in seed_run_dirs],
            "leaderboard": (output_root / "leaderboard.csv").as_posix(),
        },
    )

    for index, exp in enumerate(probe_experiments, start=1):
        time_left_hours = max(0.0, (deadline - time.time()) / 3600.0)
        if time_left_hours <= 0.25:
            logger.info("Stopping probes because less than 15 minutes remain.")
            break

        work_dir = output_root / f"{index:02d}_{exp.name}"
        work_dir.mkdir(parents=True, exist_ok=True)
        write_status(
            output_root,
            {
                "stage": "probing",
                "active_experiment": exp.name,
                "description": exp.description,
                "run_dir": work_dir.as_posix(),
                "started_at": datetime.now().isoformat(),
                "deadline": datetime.fromtimestamp(deadline).isoformat(),
                "probe_index": index,
                "probe_count": len(probe_experiments),
            },
        )

        run_training(
            gym_root=gym_root,
            workspace_root=workspace_root,
            experiment_name=exp.name,
            work_dir=work_dir,
            load_path=load_path,
            buffer_path=buffer_path,
            seed_run_dirs=seed_run_dirs,
            duration_hours=min(probe_hours, time_left_hours),
            args=args,
            overrides=exp.overrides,
        )
        plot_run(gym_root, work_dir, args.config)

        summary_path = work_dir / "summary.csv"
        ensure_exists(summary_path, f"Missing summary.csv for {exp.name}")
        summary_df = pd.read_csv(summary_path)
        metrics = score_summary(summary_df, args.score_episodes)
        metrics.update(
            {
                "experiment": exp.name,
                "description": exp.description,
                "run_dir": work_dir.as_posix(),
                "duration_minutes": min(args.probe_minutes, time_left_hours * 60.0),
                "overrides": json.dumps(list(exp.overrides)),
            }
        )
        leaderboard_rows.append(metrics)
        write_leaderboard(output_root, leaderboard_rows)
        logger.info("Completed %s with score %.2f", exp.name, metrics["score"])

    if not leaderboard_rows:
        raise RuntimeError("No experiment completed successfully.")

    leaderboard_rows.sort(key=lambda row: row["score"], reverse=True)
    champion = leaderboard_rows[0]
    champion_run_dir = Path(champion["run_dir"])
    champion_load_path = champion_run_dir / "model" / args.champion_checkpoint
    ensure_exists(champion_load_path / "policy_net.pth", f"Champion checkpoint missing under {champion_load_path}")
    if args.champion_buffer_mode == "winner_final":
        champion_buffer_path = champion_run_dir / "model" / "final"
    else:
        champion_buffer_path = buffer_path
    ensure_exists(champion_buffer_path / "replay_buffer.pkl", f"Champion replay buffer missing under {champion_buffer_path}")

    remaining_hours = max(0.0, (deadline - time.time()) / 3600.0)
    if remaining_hours > 0.05:
        champion_dir = output_root / f"champion_{champion['experiment']}"
        champion_dir.mkdir(parents=True, exist_ok=True)
        write_status(
            output_root,
            {
                "stage": "champion",
                "active_experiment": champion["experiment"],
                "description": champion["description"],
                "score": champion["score"],
                "run_dir": champion_dir.as_posix(),
                "source_run_dir": champion_run_dir.as_posix(),
                "source_checkpoint": champion_load_path.as_posix(),
                "buffer_source": champion_buffer_path.as_posix(),
                "remaining_hours": remaining_hours,
                "deadline": datetime.fromtimestamp(deadline).isoformat(),
            },
        )
        champion_overrides = tuple(json.loads(champion["overrides"]))
        run_training(
            gym_root=gym_root,
            workspace_root=workspace_root,
            experiment_name=f"champion_{champion['experiment']}",
            work_dir=champion_dir,
            load_path=champion_load_path,
            buffer_path=champion_buffer_path,
            seed_run_dirs=seed_run_dirs,
            duration_hours=remaining_hours,
            args=args,
            overrides=champion_overrides,
        )
        plot_run(gym_root, champion_dir, args.config)
        champion_summary = pd.read_csv(champion_dir / "summary.csv")
        champion_metrics = score_summary(champion_summary, args.score_episodes)
        champion_metrics.update(
            {
                "experiment": f"champion_{champion['experiment']}",
                "description": f"Continuation of {champion['experiment']}",
                "run_dir": champion_dir.as_posix(),
                "duration_minutes": remaining_hours * 60.0,
                "overrides": champion["overrides"],
            }
        )
        leaderboard_rows.append(champion_metrics)
        write_leaderboard(output_root, leaderboard_rows)
        champion = champion_metrics if champion_metrics["score"] >= champion["score"] else champion

    (output_root / "winner.json").write_text(json.dumps(champion, indent=2), encoding="utf-8")
    write_status(
        output_root,
        {
            "stage": "complete",
            "finished_at": datetime.now().isoformat(),
            "winner": champion,
            "leaderboard": (output_root / "leaderboard.csv").as_posix(),
        },
    )
    logger.info("Sweep complete. Winner: %s score %.2f", champion["experiment"], champion["score"])


if __name__ == "__main__":
    main()
