# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""G1 rough-terrain fine-tune from a flat velocity checkpoint.

Uses official G1 rough terrain, but keeps the **flat** observation space
(no height scan) so ``g1_run_to_goal`` / G1-flat checkpoints load cleanly.
"""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg


def _play_terrain(
    difficulty: str = "mid",
    num_rows: int = 1,
    num_cols: int = 1,
) -> TerrainGeneratorCfg:
    """Play terrain. Use rows×cols tiles and fill with the same number of envs
    so the video has no empty neighboring cells.

    Difficulty scales step / noise height (easy < mid < hard).
    """
    if difficulty == "easy":
        step = (0.05, 0.08)
        box_h = (0.04, 0.08)
        noise = (0.01, 0.04)
        slope = (0.0, 0.15)
    elif difficulty == "hard":
        step = (0.15, 0.23)
        box_h = (0.12, 0.20)
        noise = (0.06, 0.10)
        slope = (0.25, 0.40)
    else:  # mid
        step = (0.08, 0.15)
        box_h = (0.06, 0.12)
        noise = (0.03, 0.07)
        slope = (0.1, 0.25)

    return TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=5.0,
        num_rows=num_rows,
        num_cols=num_cols,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=False,
        sub_terrains={
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.25,
                step_height_range=step,
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
            "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.25, grid_width=0.45, grid_height_range=box_h, platform_width=2.0
            ),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.25, noise_range=noise, noise_step=0.02, border_width=0.25
            ),
            "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.25, slope_range=slope, platform_width=2.0, border_width=0.25
            ),
        },
    )


@configclass
class G1RoughEnvCfgFT(G1RoughEnvCfg):
    """Fine-tune on rough terrain with flat-compatible observations."""

    def __post_init__(self):
        super().__post_init__()

        # Match flat-policy obs dim (base g1_run_to_goal / G1 flat ckpt).
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # Widen vx for fine-tune; keep vy/wz as the base flat run (model_1749).
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # Start curriculum at easier levels; still use terrain curriculum.
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class G1RoughEnvCfgFT_PLAY(G1RoughEnvCfgFT):
    """Play / record on rough terrain."""

    difficulty: str = "mid"
    play_rows: int = 1
    play_cols: int = 1

    def __post_init__(self):
        super().__post_init__()
        n = self.play_rows * self.play_cols
        self.scene.num_envs = n
        self.scene.env_spacing = 8.0
        self.episode_length_s = 30.0

        # Fill every generated tile with a robot (no empty neighboring squares).
        self.scene.terrain.max_init_terrain_level = max(self.play_rows - 1, 0)
        self.scene.terrain.terrain_generator = _play_terrain(
            self.difficulty, num_rows=self.play_rows, num_cols=self.play_cols
        )

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # Fixed forward walk so the camera shows robots on uneven ground.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Wide shot of the pack.
        self.viewer.eye = (18.0, -18.0, 10.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0


@configclass
class G1RoughEnvCfgFT_PLAY_EASY(G1RoughEnvCfgFT_PLAY):
    difficulty: str = "easy"


@configclass
class G1RoughEnvCfgFT_PLAY_MID(G1RoughEnvCfgFT_PLAY):
    difficulty: str = "mid"


@configclass
class G1RoughEnvCfgFT_PLAY_HARD(G1RoughEnvCfgFT_PLAY):
    """Hard demo: 6×6 hard tiles, one robot each (36 total), even spawn."""

    difficulty: str = "hard"
    play_rows: int = 6
    play_cols: int = 6

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 40.0
        # No terrain-level reshuffle on timeout (keeps the even grid).
        self.curriculum.terrain_levels = None
        # Tight reset so robots stay near their tile centers.
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.2, 0.2),
            "y": (-0.2, 0.2),
            "yaw": (-0.1, 0.1),
        }
        # Pull camera farther back for the larger pack.
        self.viewer.eye = (28.0, -28.0, 16.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        self.viewer.origin_type = "world"
