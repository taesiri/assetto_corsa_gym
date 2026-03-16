import os
import sys
import argparse
import logging
import time
import json
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
import torch

sys.path.extend([os.path.abspath("./assetto_corsa_gym"), "./algorithm/discor"])

import AssettoCorsaEnv.assettoCorsa as assettoCorsa
from discor.algorithm import SAC, DisCor, SharedBackboneSAC
from discor.agent import Agent
import common.misc as misc
import common.logging_config as logging_config
from common.logger import Logger

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train AssettoCorsaGym for a fixed wall-clock duration.")
    parser.add_argument("--config", default="config.yml", type=str, help="Path to configuration file")
    parser.add_argument("--load_path", type=str, default=None, help="Model directory to resume from")
    parser.add_argument(
        "--load-buffer",
        action="store_true",
        help="Load replay_buffer.pkl from --load_path unless --buffer-path is provided",
    )
    parser.add_argument(
        "--buffer-path",
        type=str,
        default=None,
        help="Directory containing replay_buffer.pkl, or a direct path to replay_buffer.pkl",
    )
    parser.add_argument("--algo", type=str, default="sac", help="Algorithm type")
    parser.add_argument("--cuda-device", type=int, default=None, help="CUDA device index override")
    parser.add_argument("--profile-only", action="store_true", help="Emit throughput-oriented profile metrics for the run")
    parser.add_argument(
        "--offline-refresh-minutes",
        type=float,
        default=0.0,
        help="Replay-only optimizer time budget to run before the live rollout starts",
    )
    parser.add_argument("--duration-hours", type=float, default=3.0, help="Wall-clock budget in hours")
    parser.add_argument(
        "--seed-laps-dir",
        action="append",
        default=[],
        help="Directory containing prior *.parquet or *.pkl lap files to seed the replay buffer",
    )
    parser.add_argument(
        "--seed-manifest",
        action="append",
        default=[],
        help="Manifest from curate_top_laps.py; its sibling laps/ directory is loaded into the replay buffer",
    )
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Hydra-style key=value overrides")
    args = parser.parse_args()
    args.load_path = os.path.abspath(args.load_path) + os.sep if args.load_path is not None else None
    args.buffer_path = os.path.abspath(args.buffer_path) if args.buffer_path is not None else None
    args.seed_laps_dir = [os.path.abspath(path) for path in args.seed_laps_dir]
    args.seed_manifest = [os.path.abspath(path) for path in args.seed_manifest]
    return args


def make_work_dir(config):
    if config.work_dir is not None:
        work_dir = os.path.abspath(config.work_dir)
    else:
        work_dir = os.path.abspath(
            os.path.join("outputs", datetime.now().strftime("%Y%m%d_%H%M%S.%f")[:-3] + "_duration")
        )
    os.makedirs(work_dir, exist_ok=True)
    return work_dir + os.sep


def build_algo(args, env, config, device):
    use_shared_backbone = args.algo in ("shared_sac", "shared_backbone_sac") or (
        args.algo == "sac" and bool(getattr(getattr(config, "UnifiedBackbone", {}), "enabled", False))
    )
    if args.algo == "discor":
        return DisCor(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=device,
            seed=config.seed,
            **OmegaConf.to_container(config.SAC),
            **OmegaConf.to_container(config.DisCor),
        )
    if use_shared_backbone:
        return SharedBackboneSAC(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=device,
            seed=config.seed,
            unified_backbone_config=OmegaConf.to_container(config.UnifiedBackbone),
            obs_channel_names=getattr(env, "full_obs_channel_names", None) or getattr(env, "obs_enabled_channels", None),
            task_id_dim=getattr(getattr(env, "tasks_ids", None), "num_tasks", 1),
            control_delta_dim=int(getattr(config.UnifiedBackbone, "control_delta_dim", 128)),
            **OmegaConf.to_container(config.SAC),
        )
    if args.algo == "sac":
        return SAC(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=device,
            seed=config.seed,
            **OmegaConf.to_container(config.SAC),
        )
    raise ValueError("algo must be 'sac', 'shared_sac', or 'discor'")


def resolve_seed_dirs(seed_dirs, seed_manifests):
    resolved_dirs = []
    seen = set()

    def add_seed_dir(path_str):
        path = Path(path_str).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Seed path not found: {path}")
        key = path.as_posix()
        if key not in seen:
            seen.add(key)
            resolved_dirs.append(path)

    for seed_dir in seed_dirs:
        add_seed_dir(seed_dir)

    for manifest in seed_manifests:
        manifest_path = Path(manifest).resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Seed manifest not found: {manifest_path}")
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        laps_dir = manifest_path.parent / "laps"
        if laps_dir.exists():
            add_seed_dir(laps_dir)
            continue
        if manifest_payload.get("laps"):
            add_seed_dir(manifest_path.parent)
            continue
        raise FileNotFoundError(f"Could not resolve lap directory from seed manifest: {manifest_path}")

    return resolved_dirs


def select_cuda_device(config, cli_device_index):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for live training.")

    runtime_cfg = getattr(config, "Runtime", None)
    requested_index = cli_device_index
    if requested_index is None and runtime_cfg is not None:
        requested_index = int(getattr(runtime_cfg, "cuda_device_index", 0))
    if requested_index is None:
        requested_index = 0

    device_count = torch.cuda.device_count()
    if requested_index < 0 or requested_index >= device_count:
        logger.warning(
            "Requested CUDA device %s is unavailable. Falling back to cuda:0 out of %d visible devices.",
            requested_index,
            device_count,
        )
        requested_index = 0

    torch.cuda.set_device(requested_index)
    device = torch.device(f"cuda:{requested_index}")
    logger.info("Using CUDA device %d: %s", requested_index, torch.cuda.get_device_name(requested_index))
    return device, requested_index


def load_replay_buffer(agent, path):
    buffer_file = path
    if os.path.isdir(buffer_file):
        buffer_file = os.path.join(buffer_file, "replay_buffer.pkl")

    with open(buffer_file, "rb") as handle:
        agent._replay_buffer = pickle.load(handle)

    agent._steps = agent._replay_buffer._n
    logger.info("Loaded replay buffer from %s. Number of steps: %d", buffer_file, len(agent._replay_buffer))


def write_run_summary(path, started_at, duration_hours, requested_deadline, seed_dirs, seed_manifests, offline_refresh_minutes, cuda_device_index, profile_mode, agent):
    payload = {
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now().isoformat(),
        "requested_duration_hours": duration_hours,
        "requested_deadline_epoch": requested_deadline,
        "seed_laps_dirs": seed_dirs,
        "seed_manifests": seed_manifests,
        "offline_refresh_minutes": offline_refresh_minutes,
        "cuda_device_index": cuda_device_index,
        "profile_mode": profile_mode,
        "episodes": agent._episodes,
        "steps": agent._steps,
        "buffer_size": len(agent._replay_buffer),
        "best_reward": agent.best_reward,
        "best_lap_time": agent.best_lap_time,
        "optimizer_updates_total": agent._optimizer_updates_total,
        "optimizer_updates_live_total": agent._optimizer_updates_live_total,
        "optimizer_updates_post_episode_total": agent._optimizer_updates_post_episode_total,
        "optimizer_updates_offline_refresh_total": agent._optimizer_updates_offline_refresh_total,
        "optimizer_update_time_total_s": agent._optimizer_update_time_total_s,
        "reset_time_total_s": agent._reset_time_total_s,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def numeric_mean(frame, column, default=0.0):
    if frame.empty or column not in frame.columns:
        return default
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return default
    return float(values.mean())


def numeric_max(frame, column, default=0.0):
    if frame.empty or column not in frame.columns:
        return default
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return default
    return float(values.max())


def numeric_sum(frame, column, default=0.0):
    if frame.empty or column not in frame.columns:
        return default
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return default
    return float(values.sum())


def write_profile_summary(path, started_at, seed_dirs, seed_manifests, offline_refresh_minutes, cuda_device_index, summary_df, agent):
    elapsed_s = max((datetime.now() - started_at).total_seconds(), 1e-6)
    total_episode_steps = int(numeric_sum(summary_df, "ep_steps", 0.0))
    total_lap_progress = numeric_sum(summary_df, "lap_progress_laps", 0.0)
    best_lap_progress = numeric_max(summary_df, "lap_progress_laps", 0.0)
    profile_payload = {
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now().isoformat(),
        "elapsed_s": elapsed_s,
        "cuda_device_index": cuda_device_index,
        "seed_laps_dirs": [Path(path).as_posix() for path in seed_dirs],
        "seed_manifests": [Path(path).as_posix() for path in seed_manifests],
        "offline_refresh_minutes": offline_refresh_minutes,
        "episodes": int(len(summary_df)),
        "env_steps_total": total_episode_steps,
        "env_steps_per_sec_wall_clock": total_episode_steps / elapsed_s,
        "gradient_updates_total": int(agent._optimizer_updates_total),
        "gradient_updates_per_sec_wall_clock": agent._optimizer_updates_total / elapsed_s,
        "optimizer_updates_live_total": int(agent._optimizer_updates_live_total),
        "optimizer_updates_post_episode_total": int(agent._optimizer_updates_post_episode_total),
        "optimizer_updates_offline_refresh_total": int(agent._optimizer_updates_offline_refresh_total),
        "mean_step_perf_s": numeric_mean(summary_df, "step_perf_mean", 0.0),
        "mean_update_perf_s": numeric_mean(summary_df, "update_model_perf_mean", 0.0),
        "mean_action_perf_s": numeric_mean(summary_df, "action_perf_mean", 0.0),
        "mean_reset_time_s": numeric_mean(summary_df, "reset_time_s", 0.0),
        "mean_packages_lost_per_episode": numeric_mean(summary_df, "packages_lost", 0.0),
        "max_packages_lost_per_episode": numeric_max(summary_df, "packages_lost", 0.0),
        "best_lap_progress_laps": best_lap_progress,
        "best_lap_progress_laps_per_hour": best_lap_progress / max(elapsed_s / 3600.0, 1e-6),
        "total_lap_progress_laps": total_lap_progress,
        "total_lap_progress_laps_per_hour": total_lap_progress / max(elapsed_s / 3600.0, 1e-6),
        "best_completed_lap_count": numeric_max(summary_df, "completed_lap_count", 0.0),
        "best_reward": numeric_max(summary_df, "ep_reward", 0.0),
    }
    path.write_text(json.dumps(profile_payload, indent=2), encoding="utf-8")
    return profile_payload


def main():
    args = parse_args()

    config = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_dotlist(args.overrides)
    config = OmegaConf.merge(config, cli_conf)
    if "Runtime" not in config:
        config.Runtime = OmegaConf.create()
    config.Runtime.profile_mode = bool(getattr(config.Runtime, "profile_mode", False) or args.profile_only)
    config.work_dir = make_work_dir(config)
    os.environ["ACGYM_TRAINING_LOW_FI"] = "1" if bool(getattr(config.Runtime, "low_fi_training", False)) else "0"

    logging_config.create_logging(level=logging.DEBUG, file_name=config.work_dir + "log.log")
    logging.getLogger().setLevel(logging.INFO)

    misc.get_system_info()
    misc.get_git_commit_info()

    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(config))
    logger.info("work_dir: " + config.work_dir)

    env = assettoCorsa.make_ac_env(cfg=config, work_dir=config.work_dir)

    device, cuda_device_index = select_cuda_device(config, args.cuda_device)
    config.Runtime.cuda_device_index = cuda_device_index
    if getattr(config, "UnifiedBackbone", None) is not None:
        config.UnifiedBackbone.head_device_index = cuda_device_index
        config.UnifiedBackbone.ctrl_rate_hz = float(getattr(config.AssettoCorsa, "ego_sampling_freq", 25))

    algo = build_algo(args, env, config, device)

    config.exp_name = f"{config.AssettoCorsa.car}-{config.AssettoCorsa.track}"
    config.action_dim = env.action_dim

    if not config.disable_wandb:
        wandb_logger = Logger(config.copy())
    else:
        wandb_logger = None

    agent = Agent(
        env=env,
        test_env=env,
        algo=algo,
        log_dir=config.work_dir,
        device=device,
        seed=config.seed,
        wandb_logger=wandb_logger,
        save_final_buffer=True,
        **config.Agent,
    )

    if args.load_path is not None:
        agent.load(args.load_path, load_buffer=False)
    if args.load_buffer or args.buffer_path is not None:
        load_replay_buffer(agent, args.buffer_path or args.load_path)

    resolved_seed_dirs = resolve_seed_dirs(args.seed_laps_dir, args.seed_manifest)
    for seed_dir in resolved_seed_dirs:
        logger.info("Seeding replay buffer from %s", seed_dir)
        agent.load_pre_train_data(seed_dir.as_posix(), env)

    if resolved_seed_dirs:
        agent._steps = len(agent._replay_buffer)
        logger.info("Seeded replay buffer size: %d. Continuing from step counter %d.", len(agent._replay_buffer), agent._steps)

    started_at = datetime.now()
    deadline = time.time() + (args.duration_hours * 3600.0)
    if args.offline_refresh_minutes > 0:
        logger.info("Running replay-only offline refresh for %.2f minutes before live rollout.", args.offline_refresh_minutes)
        refresh_result = agent.run_replay_update_burst(
            max_seconds=args.offline_refresh_minutes * 60.0,
            max_updates=1_000_000_000,
            source="offline_refresh",
        )
        refresh_summary = {
            "updates": int(refresh_result.get("updates", 0)),
            "elapsed_s": float(refresh_result.get("elapsed_s", 0.0)),
        }
        (Path(config.work_dir) / "offline_refresh_summary.json").write_text(
            json.dumps(refresh_summary, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Offline refresh completed. updates=%d elapsed_s=%.2f",
            refresh_summary["updates"],
            refresh_summary["elapsed_s"],
        )
    logger.info("Training until wall-clock deadline in %.2f hours.", args.duration_hours)

    try:
        while time.time() < deadline:
            agent.train_episode()
    finally:
        final_dir = os.path.join(agent._model_dir, "final")
        agent.save(final_dir, save_buffer=True)
        write_run_summary(
            Path(config.work_dir) / "time_budget_summary.json",
            started_at,
            args.duration_hours,
            deadline,
            [path.as_posix() for path in resolved_seed_dirs],
            args.seed_manifest,
            args.offline_refresh_minutes,
            cuda_device_index,
            bool(config.Runtime.profile_mode),
            agent,
        )
        summary_path = Path(config.work_dir) / "summary.csv"
        summary_df = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
        write_profile_summary(
            Path(config.work_dir) / "profile_summary.json",
            started_at,
            resolved_seed_dirs,
            args.seed_manifest,
            args.offline_refresh_minutes,
            cuda_device_index,
            summary_df,
            agent,
        )
        agent._writer.close()
        if agent.wandb_logger:
            agent.wandb_logger.finish()
        logger.info("done training")


if __name__ == "__main__":
    main()
