import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.extend([os.path.abspath("./assetto_corsa_gym"), "./algorithm/discor"])

import AssettoCorsaEnv.assettoCorsa as assettoCorsa
from discor.agent import Agent
from discor.algorithm import SAC


DATASET_FILES = [
    "data_sets/ks_barcelona-layout_gp/ks_mazda_miata/20240219_044921_SAC_25hz/laps/20240219_044935.455_raw_data.pkl",
    "data_sets/ks_barcelona-layout_gp/ks_mazda_miata/20240219_044921_SAC_25hz/laps/20240219_044948.289_raw_data.pkl",
    "data_sets/ks_barcelona-layout_gp/ks_mazda_miata/20240219_044921_SAC_25hz/laps/20240219_045000.537_raw_data.pkl",
    "data_sets/ks_barcelona-layout_gp/ks_mazda_miata/20240219_044921_SAC_25hz/laps/20240219_045018.740_raw_data.pkl",
    "data_sets/ks_barcelona-layout_gp/ks_mazda_miata/20240219_044921_SAC_25hz/laps/20240219_045026.808_raw_data.pkl",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run a short offline SAC pretrain and export plots.")
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--dataset-root", default=r"C:\Workspace\RacingSim\offline_dataset")
    parser.add_argument("--updates", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--memory-size", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", default=None)
    return parser.parse_args()


def ensure_dataset(dataset_root: Path) -> Path:
    last_path = None
    for file_name in DATASET_FILES:
        last_path = hf_hub_download(
            repo_id="dasgringuen/assettoCorsaGym",
            repo_type="dataset",
            filename=file_name,
            local_dir=dataset_root.as_posix(),
        )
        print(f"downloaded {file_name}")
    if last_path is None:
        raise RuntimeError("No dataset files configured.")
    return Path(last_path).parent


def build_config(config_path: str, work_dir: Path, seed: int, batch_size: int, memory_size: int):
    cfg = OmegaConf.load(config_path)
    cfg.seed = seed
    cfg.disable_wandb = True
    cfg.work_dir = work_dir.as_posix()
    cfg.AssettoCorsa.screen_capture_enable = False
    cfg.AssettoCorsa.track = "ks_barcelona-layout_gp"
    cfg.AssettoCorsa.car = "ks_mazda_miata"
    cfg.Agent.batch_size = batch_size
    cfg.Agent.memory_size = memory_size
    cfg.Agent.start_steps = 0
    cfg.Agent.eval_interval = 0
    cfg.Agent.log_interval = 1
    cfg.SAC.log_interval = 1
    return cfg


def create_agent(cfg):
    env = assettoCorsa.make_ac_env(cfg=cfg, work_dir=cfg.work_dir)
    original_get_reward = env.get_reward

    def offline_safe_reward(state, actions_diff):
        if "dist_to_border" not in state and "reward" in state:
            return np.array([state["reward"]], dtype=np.float32)
        return original_get_reward(state, actions_diff)

    env.get_reward = offline_safe_reward
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this offline pretrain run.")
    device = torch.device("cuda")
    algo = SAC(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device=device,
        seed=cfg.seed,
        **OmegaConf.to_container(cfg.SAC),
    )
    agent = Agent(
        env=env,
        test_env=env,
        algo=algo,
        log_dir=cfg.work_dir,
        device=device,
        seed=cfg.seed,
        num_steps=0,
        batch_size=cfg.Agent.batch_size,
        memory_size=cfg.Agent.memory_size,
        update_interval=1,
        start_steps=0,
        log_interval=1,
        eval_interval=0,
        num_eval_episodes=1,
        checkpoint_freq=0,
        use_offline_buffer=False,
        offline_buffer_size=cfg.Agent.offline_buffer_size,
        save_final_buffer=False,
        wandb_logger=None,
    )
    return env, agent


def run_updates(agent: Agent, laps_dir: Path, updates: int):
    agent.load_pre_train_data(laps_dir.as_posix(), agent._env)
    buffer_size = len(agent._replay_buffer)
    if buffer_size < agent._batch_size:
        raise RuntimeError(f"Replay buffer too small for batch size: {buffer_size} < {agent._batch_size}")

    policy_metrics = []
    for idx in range(updates):
        stats = agent.update_model()
        if stats:
            stats = {"update": idx + 1, **stats}
            policy_metrics.append(stats)
        if (idx + 1) % 25 == 0:
            print(f"completed {idx + 1}/{updates} updates")

    return buffer_size, pd.DataFrame(policy_metrics)


def load_scalar_series(summary_dir: Path):
    accumulator = EventAccumulator(summary_dir.as_posix())
    accumulator.Reload()
    frames = []
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        frames.append(
            pd.DataFrame(
                {
                    "tag": tag,
                    "step": [event.step for event in events],
                    "value": [event.value for event in events],
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["tag", "step", "value"])
    return pd.concat(frames, ignore_index=True)


def plot_training_metrics(scalars: pd.DataFrame, output_file: Path):
    tags = [
        "loss/policy",
        "loss/Q",
        "stats/alpha",
        "stats/entropy",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, tag in zip(axes.flat, tags):
        tag_df = scalars[scalars["tag"] == tag]
        if tag_df.empty:
            ax.set_visible(False)
            continue
        ax.plot(tag_df["step"], tag_df["value"], linewidth=2)
        ax.set_title(tag)
        ax.set_xlabel("update")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=160)
    plt.close(fig)


def plot_trajectory(env, first_file: Path, output_file: Path):
    trajectory, _ = env.load_history(first_file)
    trajectory_df = pd.DataFrame(trajectory)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(env.track.left_border_x, env.track.left_border_y, color="black", linewidth=1, label="left border")
    ax.plot(env.track.right_border_x, env.track.right_border_y, color="black", linewidth=1, label="right border")
    ax.plot(trajectory_df["world_position_x"], trajectory_df["world_position_y"], color="#d1495b", linewidth=2, label="offline lap")
    ax.set_title("Offline Lap Overlay")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_file, dpi=160)
    plt.close(fig)


def write_summary(output_file: Path, work_dir: Path, laps_dir: Path, buffer_size: int, updates: int, metrics_df: pd.DataFrame, scalars_df: pd.DataFrame):
    summary = {
        "work_dir": work_dir.as_posix(),
        "laps_dir": laps_dir.as_posix(),
        "buffer_size": int(buffer_size),
        "updates": int(updates),
        "final_policy_loss": None if metrics_df.empty else float(metrics_df["policy_loss"].iloc[-1]),
        "final_entropy": None if metrics_df.empty else float(metrics_df["entropy"].iloc[-1]),
        "scalar_tags": sorted(scalars_df["tag"].unique().tolist()) if not scalars_df.empty else [],
    }
    output_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    run_name = datetime.now().strftime("offline_pretrain_%Y%m%d_%H%M%S")
    work_dir = Path(args.work_dir) if args.work_dir else Path("outputs") / run_name
    work_dir.mkdir(parents=True, exist_ok=True)

    laps_dir = ensure_dataset(Path(args.dataset_root))
    cfg = build_config(args.config, work_dir.resolve(), args.seed, args.batch_size, args.memory_size)
    env, agent = create_agent(cfg)

    try:
        buffer_size, metrics_df = run_updates(agent, laps_dir, args.updates)
        checkpoint_dir = Path(agent._model_dir) / "offline_pretrain_final"
        agent.save(checkpoint_dir.as_posix(), save_buffer=False)
        agent._writer.flush()

        scalars_df = load_scalar_series(Path(agent._summary_dir))
        scalars_df.to_csv(work_dir / "offline_scalars.csv", index=False)
        metrics_df.to_csv(work_dir / "offline_policy_metrics.csv", index=False)
        plot_training_metrics(scalars_df, work_dir / "offline_training_metrics.png")
        plot_trajectory(env, sorted(laps_dir.glob("*.pkl"))[0], work_dir / "offline_trajectory_overlay.png")
        write_summary(work_dir / "offline_run_summary.json", work_dir.resolve(), laps_dir.resolve(), buffer_size, args.updates, metrics_df, scalars_df)
        print(f"offline pretrain finished: {work_dir.resolve().as_posix()}")
    finally:
        agent._writer.close()
        env.close()


if __name__ == "__main__":
    main()
