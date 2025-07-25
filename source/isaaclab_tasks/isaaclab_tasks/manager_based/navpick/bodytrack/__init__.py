# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Humanoid locomotion environment (similar to OpenAI Gym Humanoid-v2).
"""

import gymnasium as gym

from . import agents
from .. import base

##
# Register Gym environments.
##

gym.register(
    id="bodytrack-train",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bodytrack_env_cfg:G1BodyTrackEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1BodyTrackPPORunnerCfg",
    },
)

gym.register(
    id="bodytrack-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bodytrack_env_cfg:G1BodyTrackEnvCfgPlay",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1BodyTrackPPORunnerCfg",
    },
)