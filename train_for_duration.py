import os
import sys
import argparse
import logging
import time
import json
import pickle
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf
import torch

sys.path.extend([os.path.abspath("./assetto_corsa_gym"), "./algorithm/discor"])

import AssettoCorsaEnv.assettoCorsa as assettoCorsa
from discor.algorithm import SAC, DisCor
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
    parser.add_argument("--duration-hours", type=float, default=3.0, help="Wall-clock budget in hours")
    parser.add_argument(
        "--seed-laps-dir",
        action="append",
        default=[],
        help="Directory containing prior *.parquet or *.pkl lap files to seed the replay buffer",
    )
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Hydra-style key=value overrides")
    args = parser.parse_args()
    args.load_path = os.path.abspath(args.load_path) + os.sep if args.load_path is not None else None
    args.buffer_path = os.path.abspath(args.buffer_path) if args.buffer_path is not None else None
    args.seed_laps_dir = [os.path.abspath(path) for path in args.seed_laps_dir]
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
    if args.algo == "discor":
        return DisCor(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=device,
            seed=config.seed,
            **OmegaConf.to_container(config.SAC),
            **OmegaConf.to_container(config.DisCor),
        )
    if args.algo == "sac":
        return SAC(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=device,
            seed=config.seed,
            **OmegaConf.to_container(config.SAC),
        )
    raise ValueError("algo must be 'sac' or 'discor'")


def load_replay_buffer(agent, path):
    buffer_file = path
    if os.path.isdir(buffer_file):
        buffer_file = os.path.join(buffer_file, "replay_buffer.pkl")

    with open(buffer_file, "rb") as handle:
        agent._replay_buffer = pickle.load(handle)

    agent._steps = agent._replay_buffer._n
    logger.info("Loaded replay buffer from %s. Number of steps: %d", buffer_file, len(agent._replay_buffer))


def write_run_summary(path, started_at, duration_hours, requested_deadline, seed_dirs, agent):
    payload = {
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now().isoformat(),
        "requested_duration_hours": duration_hours,
        "requested_deadline_epoch": requested_deadline,
        "seed_laps_dirs": seed_dirs,
        "episodes": agent._episodes,
        "steps": agent._steps,
        "buffer_size": len(agent._replay_buffer),
        "best_reward": agent.best_reward,
        "best_lap_time": agent.best_lap_time,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    args = parse_args()

    config = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_dotlist(args.overrides)
    config = OmegaConf.merge(config, cli_conf)
    config.work_dir = make_work_dir(config)

    logging_config.create_logging(level=logging.DEBUG, file_name=config.work_dir + "log.log")
    logging.getLogger().setLevel(logging.INFO)

    misc.get_system_info()
    misc.get_git_commit_info()

    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(config))
    logger.info("work_dir: " + config.work_dir)

    env = assettoCorsa.make_ac_env(cfg=config, work_dir=config.work_dir)

    device = torch.device("cuda")
    assert device.type == "cuda", "Only cuda is supported"

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

    for seed_dir in args.seed_laps_dir:
        logger.info("Seeding replay buffer from %s", seed_dir)
        agent.load_pre_train_data(seed_dir, env)

    if args.seed_laps_dir:
        agent._steps = len(agent._replay_buffer)
        logger.info("Seeded replay buffer size: %d. Continuing from step counter %d.", len(agent._replay_buffer), agent._steps)

    started_at = datetime.now()
    deadline = time.time() + (args.duration_hours * 3600.0)
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
            args.seed_laps_dir,
            agent,
        )
        agent._writer.close()
        if agent.wandb_logger:
            agent.wandb_logger.finish()
        logger.info("done training")


if __name__ == "__main__":
    main()
