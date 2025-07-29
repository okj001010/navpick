# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from ...base.agents.rsl_rl_ppo_cfg import G1BasePPORunnerCfg


@configclass
class G1BodyPosePPORunnerCfg(G1BasePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "g1_bodypose"