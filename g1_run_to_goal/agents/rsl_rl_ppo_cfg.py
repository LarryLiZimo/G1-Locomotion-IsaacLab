# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""PPO cfg — same as official G1 flat; logs under ``g1_run_to_goal``."""

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1FlatPPORunnerCfg,
)


@configclass
class G1RunToGoalPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        # Fine-tune with widened vx; separate log dir from the base run.
        self.experiment_name = "g1_wide_vx"
        self.max_iterations = 1000
