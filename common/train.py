#!/usr/bin/env python3
"""Minimal RSL-RL PPO trainer. Run from a task directory (logs/videos land here)."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

TASK_ROOT = Path.cwd()
ASSIGNMENT_ROOT = TASK_ROOT.parent
ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_PATH", "/data/Isaac-platform/IsaacLab"))
sys.path.insert(0, str(ASSIGNMENT_ROOT))
sys.path.insert(0, str(ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"))

from isaaclab.app import AppLauncher

import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Train PPO (task-local logs).")
parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--video_length", type=int, default=150)
parser.add_argument("--video_interval", type=int, default=3200)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

try:
    import humanoid_run_to_goal  # noqa: F401
except ImportError:
    pass
try:
    import g1_run_to_goal  # noqa: F401
except ImportError:
    pass
try:
    import g1_rough  # noqa: F401
except ImportError:
    pass

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root = Path(TASK_ROOT, "logs", "rsl_rl", agent_cfg.experiment_name).resolve()
    log_dir = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir = Path(f"{log_dir}_{agent_cfg.run_name}")
    env_cfg.log_dir = str(log_dir)
    print(f"[INFO] Logging -> {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    video_folder = Path(TASK_ROOT, "videos", "train")
    video_folder.mkdir(parents=True, exist_ok=True)
    if args_cli.video:
        video_kwargs = {
            "video_folder": str(video_folder),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "name_prefix": f"run-{log_dir.name}",
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)

    if agent_cfg.resume:
        from isaaclab.utils.assets import retrieve_file_path
        from isaaclab_tasks.utils import get_checkpoint_path

        # Allow resuming from an explicit file path (cross-experiment fine-tune).
        ckpt_arg = getattr(args_cli, "checkpoint", None)
        if ckpt_arg and (os.path.isfile(ckpt_arg) or "/" in str(ckpt_arg) or str(ckpt_arg).endswith(".pt")):
            resume_path = retrieve_file_path(ckpt_arg)
        else:
            resume_path = get_checkpoint_path(str(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Resume from {resume_path}")
        runner.load(resume_path)

    dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
