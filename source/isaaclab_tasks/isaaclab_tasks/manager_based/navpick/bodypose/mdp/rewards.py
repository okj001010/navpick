# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, cast

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from .commands.bodypose_command import BodyPoseCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# FIXME(OKJ): track body command
def body_pose_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "body_pose",
    left_hip_pitch_joint: str = "left_hip_pitch_joint",
    right_hip_pitch_joint: str = "right_hip_pitch_joint",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for body-pose adjustment."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = cast(BodyPoseCommand, env.command_manager.get_term(command_name))
    goal_base_height, goal_hip_pitch = command.command[:, 0], command.command[:, 1]
    
    left_hip_pitch_joint_id = asset.find_joints(left_hip_pitch_joint)[0][0]
    right_hip_pitch_joint_id = asset.find_joints(right_hip_pitch_joint)[0][0]
    curr_base_height = asset.data.root_state_w[:, 2]
    curr_left_hip_pitch = asset.data.joint_pos[:, left_hip_pitch_joint_id]
    curr_right_hip_pitch = asset.data.joint_pos[:, right_hip_pitch_joint_id]
    
    height_rew = torch.exp(-4.0 * torch.square(curr_base_height - goal_base_height))
    hip_pitch_rew = torch.exp(-4.0 * torch.square(curr_left_hip_pitch - goal_hip_pitch) +
                              -4.0 * torch.square(curr_right_hip_pitch - goal_hip_pitch))
    # TODO(OKJ): tune the weights
    return height_rew + hip_pitch_rew


def waist_roll_error(
    env: ManagerBasedRLEnv,
    alpha: float = 1.0,
    waist_pitch_joint_name: str = "waist_pitch_joint",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for waist roll error."""
    asset: Articulation = env.scene[asset_cfg.name]
    waist_pitch_joint_id = asset.find_joints(waist_pitch_joint_name)[0][0]
    waist_pitch = asset.data.joint_pos[:, waist_pitch_joint_id]
    return torch.exp(-alpha * torch.square(waist_pitch))


def leg_pos_symmetry(
    env: ManagerBasedRLEnv,
    left_leg_joint_names: str,
    right_leg_joint_names: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for leg position symmetry."""
    asset: Articulation = env.scene[asset_cfg.name]
    left_leg_joint_ids = asset.find_joints(left_leg_joint_names)[0]
    right_leg_joint_ids = asset.find_joints(right_leg_joint_names)[0]
    
    left_leg_joint = asset.data.joint_pos[:, left_leg_joint_ids]
    right_leg_joint = asset.data.joint_pos[:, right_leg_joint_ids]

    return torch.norm(left_leg_joint - right_leg_joint)