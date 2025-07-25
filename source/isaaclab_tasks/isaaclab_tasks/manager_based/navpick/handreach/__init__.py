# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Humanoid locomotion environment (similar to OpenAI Gym Humanoid-v2).
"""

import gymnasium as gym

from . import agents

# Joint names for the hand reach task
handreach_joint_names = [
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint',
    'right_wrist_pitch_joint',
    'right_wrist_yaw_joint'
]


##
# Register Gym environments.
##

gym.register(
    id="handreach-train",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.handreach_env_cfg:G1HandReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1HandReachPPORunnerCfg",
    },
)

gym.register(
    id="handreach-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.handreach_env_cfg:G1HandReachEnvCfgPlay",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1HandReachPPORunnerCfg",
    },
)