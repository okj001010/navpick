# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Humanoid locomotion environment (similar to OpenAI Gym Humanoid-v2).
"""

import gymnasium as gym

from . import agents
from ..base import base_joint_names

# Joint names for the hand reach task
bodypose_joint_names = [
    'left_hip_pitch_joint',
    'right_hip_pitch_joint',
    'waist_yaw_joint',
    'left_hip_roll_joint',
    'right_hip_roll_joint',
    'waist_roll_joint',
    'left_hip_yaw_joint',
    'right_hip_yaw_joint',
    'waist_pitch_joint',
    'left_knee_joint',
    'right_knee_joint',
    'left_ankle_pitch_joint',
    'right_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_ankle_roll_joint',
]

# Joint names for non-hand reach task
non_bodypose_joint_names = [
    joint for joint in base_joint_names if joint not in bodypose_joint_names
]

##
# Register Gym environments.
##

gym.register(
    id="bodypose-train",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bodypose_env_cfg:G1BodyPoseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1BodyPosePPORunnerCfg",
    },
)

gym.register(
    id="bodypose-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bodypose_env_cfg:G1BodyPoseEnvCfgPlay",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1BodyPosePPORunnerCfg",
    },
)