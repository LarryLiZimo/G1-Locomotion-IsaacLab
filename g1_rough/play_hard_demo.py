#!/usr/bin/env python3
"""Hard spectacular play: even 1-robot-per-tile spawn, multiple long clips.

Isaac Lab's curriculum origin sampler picks terrain *rows* at random, so many
envs can land on the same cell. After ``gym.make``, we remap origins so env i
owns tile (i // cols, i % cols) uniquely.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

TASK_ROOT = Path.cwd()
ASSIGNMENT_ROOT = TASK_ROOT.parent
ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_PATH", "/data/Isaac-platform/IsaacLab"))
sys.path.insert(0, str(ASSIGNMENT_ROOT))
sys.path.insert(0, str(ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"))

from isaaclab.app import AppLauncher

import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Even-grid hard rough play (multi-clip).")
parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--video_length", type=int, default=900)
parser.add_argument("--video_name_prefix", type=str, default="play-rough-2100-hard-36")
parser.add_argument("--num_clips", type=int, default=3)
parser.add_argument("--num_envs", type=int, default=36)
parser.add_argument("--task", type=str, default="Isaac-G1-Rough-Play-Hard-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=42)
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
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

try:
    import g1_rough  # noqa: F401
except ImportError:
    pass


def _even_tile_origins(env) -> None:
    """Assign env i → unique terrain tile (row, col) covering the full grid."""
    raw = env.unwrapped
    terrain = raw.scene.terrain
    origins = terrain.terrain_origins
    if origins is None:
        print("[WARN] No terrain_origins; skipping even remap.")
        return
    num_rows, num_cols = origins.shape[:2]
    n = raw.num_envs
    if n > num_rows * num_cols:
        raise ValueError(f"num_envs={n} > tiles={num_rows * num_cols}")
    rows = torch.arange(n, device=origins.device) // num_cols
    cols = torch.arange(n, device=origins.device) % num_cols
    terrain.terrain_levels = rows.to(torch.long)
    terrain.terrain_types = cols.to(torch.long)
    terrain.env_origins[:] = origins[rows, cols]
    # Keep scene env_origins in sync if duplicated.
    if hasattr(raw.scene, "env_origins"):
        raw.scene.env_origins[:] = terrain.env_origins
    print(f"[INFO] Even spawn: {n} envs on {num_rows}x{num_cols} tiles")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root = Path(TASK_ROOT, "logs", "rsl_rl", agent_cfg.experiment_name).resolve()
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(str(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Checkpoint: {resume_path}")

    video_folder = Path(TASK_ROOT, "videos", "play")
    video_folder.mkdir(parents=True, exist_ok=True)

    for clip in range(args_cli.num_clips):
        seed = args_cli.seed + clip
        env_cfg.seed = seed
        agent_cfg.seed = seed
        print(f"======== clip {clip + 1}/{args_cli.num_clips} seed={seed} ========")

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        _even_tile_origins(env)
        env.reset()

        if args_cli.video:
            name_prefix = f"{args_cli.video_name_prefix}-clip{clip:02d}"
            video_kwargs = {
                "video_folder": str(video_folder),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "name_prefix": name_prefix,
                "disable_logger": True,
            }
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # Ensure even tiles after wrapper stack, then reset robots onto them.
        _even_tile_origins(env)
        env.unwrapped.reset()
        _even_tile_origins(env)
        env.unwrapped.reset()
        obs = env.get_observations()

        timestep = 0
        while simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
            if args_cli.video:
                timestep += 1
                if timestep >= args_cli.video_length:
                    break

        env.close()
        print(f"[INFO] Clip {clip + 1} done -> {video_folder}")

    print(f"[INFO] All clips -> {video_folder}")


if __name__ == "__main__":
    main()
    simulation_app.close()
