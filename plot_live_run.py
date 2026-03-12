import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.extend([os.path.abspath("./assetto_corsa_gym")])

import AssettoCorsaEnv.assettoCorsa as assettoCorsa


def parse_args():
    parser = argparse.ArgumentParser(description="Generate plots for a live AssettoCorsaGym run.")
    parser.add_argument("--run-dir", required=True, help="Path to a live training output directory.")
    parser.add_argument("--config", default="config.yml", help="Base config file for loading track metadata.")
    return parser.parse_args()


def load_scalar_series(summary_dir: Path) -> pd.DataFrame:
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


def plot_training_metrics(summary_df: pd.DataFrame, scalars_df: pd.DataFrame, output_file: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(summary_df["ep_count"], summary_df["ep_reward"], marker="o", linewidth=2, color="#1f77b4")
    axes[0, 0].set_title("Episode Reward")
    axes[0, 0].set_xlabel("episode")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(summary_df["ep_count"], summary_df["ep_steps"], marker="o", linewidth=2, color="#2a9d8f")
    axes[0, 1].set_title("Episode Steps")
    axes[0, 1].set_xlabel("episode")
    axes[0, 1].grid(True, alpha=0.3)

    if not scalars_df.empty and "loss/policy" in scalars_df["tag"].values:
        policy_df = scalars_df[scalars_df["tag"] == "loss/policy"]
        axes[1, 0].plot(policy_df["step"], policy_df["value"], linewidth=2, color="#d1495b")
        axes[1, 0].set_title("Policy Loss")
        axes[1, 0].set_xlabel("training step")
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].axis("off")

    axes[1, 1].plot(summary_df["ep_count"], summary_df["speed_mean"], marker="o", linewidth=2, color="#f4a261")
    axes[1, 1].set_title("Mean Speed")
    axes[1, 1].set_xlabel("episode")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_file, dpi=160)
    plt.close(fig)


def load_env(run_dir: Path, config_path: str):
    cfg = OmegaConf.load(config_path)
    static_info = json.loads((run_dir / "laps" / "static_info.json").read_text(encoding="utf-8"))
    cfg.AssettoCorsa.screen_capture_enable = False
    cfg.AssettoCorsa.track = static_info["TrackFullName"]
    cfg.AssettoCorsa.car = static_info["CarName"]
    return assettoCorsa.make_ac_env(cfg=cfg, work_dir=run_dir.as_posix())


def plot_trajectory_overlay(env, run_dir: Path, output_file: Path):
    lap_files = sorted((run_dir / "laps").glob("*_states.parquet"))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(env.track.left_border_x, env.track.left_border_y, color="black", linewidth=1)
    ax.plot(env.track.right_border_x, env.track.right_border_y, color="black", linewidth=1)

    for idx, lap_file in enumerate(lap_files, start=1):
        lap_df = pd.read_parquet(lap_file, engine="pyarrow")
        ax.plot(
            lap_df["world_position_x"],
            lap_df["world_position_y"],
            linewidth=1.5,
            label=f"episode {idx}",
        )

    ax.set_title("Live Trajectory Overlay")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    if lap_files:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_file, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()

    summary_df = pd.read_csv(run_dir / "summary.csv")
    scalars_df = load_scalar_series(run_dir / "summary")
    scalars_df.to_csv(run_dir / "live_scalars.csv", index=False)
    plot_training_metrics(summary_df, scalars_df, run_dir / "live_training_metrics.png")

    env = load_env(run_dir, args.config)
    try:
        plot_trajectory_overlay(env, run_dir, run_dir / "live_trajectory_overlay.png")
    finally:
        env.close()

    summary_payload = {
        "run_dir": run_dir.as_posix(),
        "episodes": int(len(summary_df)),
        "total_steps": int(summary_df["total_steps"].max()),
        "best_episode_reward": float(summary_df["ep_reward"].max()),
        "mean_episode_reward": float(summary_df["ep_reward"].mean()),
        "max_speed": float(summary_df["speed_max"].max()),
    }
    (run_dir / "live_run_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
