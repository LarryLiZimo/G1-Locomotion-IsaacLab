# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""G1 run-to-goal.

Training: official flat velocity tracking (random vx, vy, wz).
Inference play-to-goal: same policy; command term converts a 2D goal → velocity.
"""

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
    G1FlatEnvCfg_PLAY,
)

from .mdp.commands import GoalVelocityCommandCfg


@configclass
class G1RunToGoalEnvCfg(G1FlatEnvCfg):
    """Train velocity tracking; vx range widened for OOD, vy/wz shrunk."""

    def __post_init__(self):
        super().__post_init__()
        # Official flat was vx∈[0,1], vy∈[-0.5,0.5], wz∈[-1,1].
        # Widen forward speed; keep side/yaw random but narrower.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)


@configclass
class G1RunToGoalEnvCfg_PLAY(G1FlatEnvCfg_PLAY):
    """Play velocity tracking (random / overridden commands)."""

    pass


@configclass
class G1RunToGoalEnvCfg_GOAL_PLAY(G1FlatEnvCfg_PLAY):
    """Inference: freeze the trained policy, steer it with goal → velocity PD.

    Still named ``base_velocity`` so observations match training
    (policy still receives a 3-D velocity command).
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 8.0
        self.episode_length_s = 20.0
        self.viewer.eye = (8.0, -8.0, 4.0)
        self.viewer.lookat = (2.0, 0.0, 0.8)
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0

        # Replace random velocity sampling with goal tracking.
        self.commands.base_velocity = GoalVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(20.0, 20.0),  # one goal for the whole clip
            debug_vis=True,
            stand_distance=0.35,
            lin_gain=1.0,
            ang_gain=1.5,
            max_vel_x=1.0,
            max_vel_y=0.5,
            max_vel_yaw=1.0,
            ranges=GoalVelocityCommandCfg.Ranges(pos_x=(3.0, 3.0), pos_y=(0.0, 0.0)),
        )
