# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
import torch
from typing import TYPE_CHECKING, cast

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from .commands.bodypose_command import BodyPoseCommand
from .. import bodypose_joint_names

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def body_pose_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    hip_pitch_joint: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for body-pose adjustment."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = cast(BodyPoseCommand, env.command_manager.get_term(command_name))
    goal_base_height, goal_hip_pitch = command.command[:, 0], command.command[:, 1]
    
    hip_pitch_joint_id = asset.find_joints(hip_pitch_joint)[0]
    curr_base_height = asset.data.root_state_w[:, 2]
    curr_hip_pitch = asset.data.joint_pos[:, hip_pitch_joint_id]

    height_rew = torch.exp(-4.0 * torch.square(curr_base_height - goal_base_height))
    hip_pitch_rew = torch.exp(-4.0 * torch.sum(torch.square(curr_hip_pitch - goal_hip_pitch.unsqueeze(-1)), dim=-1))
    
    # TODO(OKJ): tune the weights
    return height_rew + hip_pitch_rew


def waist_roll_error(
    env: ManagerBasedRLEnv,
    waist_pitch_joint_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for waist roll error."""
    asset: Articulation = env.scene[asset_cfg.name]
    waist_pitch_joint_id = asset.find_joints(waist_pitch_joint_name)[0][0]
    waist_pitch = asset.data.joint_pos[:, waist_pitch_joint_id]
    return torch.exp(-4.0 * torch.square(waist_pitch))


def leg_pos_symmetry(
    env: ManagerBasedRLEnv,
    left_leg_joint_names: str,
    right_leg_joint_names: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for leg position symmetry."""
    asset: Articulation = env.scene[asset_cfg.name]
    left_leg_joint_ids, left_leg_names = asset.find_joints(left_leg_joint_names)
    right_leg_joint_ids, _ = asset.find_joints(right_leg_joint_names)
    
    left_leg_joint = asset.data.joint_pos[:, left_leg_joint_ids]
    right_leg_joint = asset.data.joint_pos[:, right_leg_joint_ids]

    # NOTE(OKJ): hip_roll, hip_yaw, and ankle_roll joints are mirrored
    mirror_list = [ -1.0 if (("roll" in n) or ("yaw" in n)) else 1.0 for n in left_leg_names]
    mirror = torch.tensor(mirror_list, device=left_leg_joint.device).unsqueeze(0)

    return torch.norm(left_leg_joint - right_leg_joint * mirror, dim=-1)


def leg_force_symmetry(
    env: ManagerBasedRLEnv,
    left_leg_joint_names: str,
    right_leg_joint_names: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for leg force symmetry."""
    asset: Articulation = env.scene[asset_cfg.name]
    left_leg_joint_ids, _ = asset.find_joints(left_leg_joint_names)
    right_leg_joint_ids, _ = asset.find_joints(right_leg_joint_names)
    
    return torch.norm(
        asset.data.applied_torque[:, left_leg_joint_ids] - asset.data.applied_torque[:, right_leg_joint_ids],
        p=1, dim=-1
    )
    

def contact_ground(
    env: ManagerBasedRLEnv,
    max_force: float,
    left_contact_link_name: str,
    right_contact_link_name: str,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Reward for ground contact: product of normalized left and right foot contact forces."""
    # extract sensors
    contact_sensors = cast(ContactSensor, env.scene[sensor_cfg.name])
    left_contact_link_id = contact_sensors.find_bodies(left_contact_link_name)[0][0]
    right_contact_link_id = contact_sensors.find_bodies(right_contact_link_name)[0][0]

    # get net contact forces (N, 3)
    left_forces = cast(torch.Tensor, contact_sensors.data.net_forces_w)[:, left_contact_link_id]
    right_forces = cast(torch.Tensor, contact_sensors.data.net_forces_w)[:, right_contact_link_id]
    
    # compute force normal (z-axis) components
    left_contact_normal = left_forces[:, 2]
    right_contact_normal = right_forces[:, 2]
    
    # normalize forces and clamp to [0, 1]
    left_contact = torch.clamp(left_contact_normal / max_force, min=0.0, max= 1.0)
    right_contact = torch.clamp(right_contact_normal / max_force, min=0.0, max= 1.0)

    return left_contact * right_contact