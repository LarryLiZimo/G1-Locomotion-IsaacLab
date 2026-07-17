# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Inference-time command: convert a 2D goal into a velocity command (vx, vy, wz).

The trained G1 policy still sees the same 3-D velocity command as in training.
This term only changes *how* that command is produced (goal PD → velocity).
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, wrap_to_pi, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class GoalVelocityCommand(CommandTerm):
    """Sample a world-frame XY goal, then emit base-frame velocity commands toward it.

    Each step:
    1. Express the goal in the robot base (yaw) frame → (dx, dy).
    2. If close enough → command (0, 0, 0) so the policy stands still.
    3. Else → proportional control clipped to the G1 training ranges
       vx∈[0, max_vel_x], vy∈[-max_vel_y, max_vel_y], wz∈[-max_vel_yaw, max_vel_yaw].
    """

    cfg: "GoalVelocityCommandCfg"

    def __init__(self, cfg: "GoalVelocityCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["dist_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reached"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            "GoalVelocityCommand:\n"
            f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tResampling time range: {self.cfg.resampling_time_range}\n"
            f"\tStand distance: {self.cfg.stand_distance}"
        )

    @property
    def command(self) -> torch.Tensor:
        """Base-frame velocity command (vx, vy, wz). Same interface as training."""
        return self.vel_command_b

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.goal_pos_w[env_ids] = self._env.scene.env_origins[env_ids]
        self.goal_pos_w[env_ids, 0] += r.uniform_(*self.cfg.ranges.pos_x)
        self.goal_pos_w[env_ids, 1] += r.uniform_(*self.cfg.ranges.pos_y)
        self.goal_pos_w[env_ids, 2] = self.robot.data.default_root_state[env_ids, 2]

    def _update_command(self):
        # Goal relative to robot, in base yaw frame.
        to_goal_w = self.goal_pos_w - self.robot.data.root_pos_w[:, :3]
        to_goal_b = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), to_goal_w)
        dx = to_goal_b[:, 0]
        dy = to_goal_b[:, 1]
        dist = torch.linalg.norm(to_goal_b[:, :2], dim=1)

        heading_err = wrap_to_pi(torch.atan2(dy, dx))

        vx = torch.clamp(self.cfg.lin_gain * dx, min=0.0, max=self.cfg.max_vel_x)
        vy = torch.clamp(self.cfg.lin_gain * dy, min=-self.cfg.max_vel_y, max=self.cfg.max_vel_y)
        wz = torch.clamp(self.cfg.ang_gain * heading_err, min=-self.cfg.max_vel_yaw, max=self.cfg.max_vel_yaw)

        # Near the goal → stand still (matches the 0-velocity standing skill from training).
        near = dist < self.cfg.stand_distance
        vx = torch.where(near, torch.zeros_like(vx), vx)
        vy = torch.where(near, torch.zeros_like(vy), vy)
        wz = torch.where(near, torch.zeros_like(wz), wz)

        self.vel_command_b[:, 0] = vx
        self.vel_command_b[:, 1] = vy
        self.vel_command_b[:, 2] = wz

        self.metrics["dist_xy"][:] = dist
        self.metrics["reached"][:] = near.float()

    def _update_metrics(self):
        # Metrics already updated in _update_command.
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
            self.goal_visualizer.set_visibility(True)
        elif hasattr(self, "goal_visualizer"):
            self.goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        zeros = torch.zeros(self.num_envs, device=self.device)
        self.goal_visualizer.visualize(
            translations=self.goal_pos_w,
            orientations=quat_from_euler_xyz(zeros, zeros, zeros),
        )


@configclass
class GoalVelocityCommandCfg(CommandTermCfg):
    """Config for :class:`GoalVelocityCommand`."""

    class_type: type = GoalVelocityCommand

    asset_name: str = "robot"

    stand_distance: float = 0.35
    """XY distance (m) below which the velocity command is zeroed."""

    lin_gain: float = 1.0
    """Proportional gain from base-frame position error to linear velocity."""

    ang_gain: float = 1.5
    """Proportional gain from heading-to-goal error to yaw rate."""

    max_vel_x: float = 1.0
    max_vel_y: float = 0.5
    max_vel_yaw: float = 1.0
    """Clip commands to the official G1 flat training ranges."""

    @configclass
    class Ranges:
        pos_x: tuple[float, float] = (2.0, 5.0)
        pos_y: tuple[float, float] = (-2.0, 2.0)

    ranges: Ranges = Ranges()

    goal_visualizer_cfg: object = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/goal")

    def __post_init__(self):
        # Larger marker so the goal is obvious in videos.
        self.goal_visualizer_cfg.markers["arrow"].scale = (0.3, 0.3, 1.2)
