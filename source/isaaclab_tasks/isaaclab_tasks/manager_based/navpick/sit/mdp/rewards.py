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
from . import zmp_rwd_computation_helper as zmp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def keep_torso_upright(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat torso orientation using L2 squared kernel.
    This is computed by penalizing the xy-components of the projected gravity vector.
    Referenced from mdp.flat_orientation_l2
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    torso_link_idx = asset.find_bodies("torso_link")[0][0]
    torso_quat_w = asset.data.body_quat_w[:, torso_link_idx, :]

    torso_projected_gravity_b = math_utils.quat_rotate_inverse(torso_quat_w, asset.data.GRAVITY_VEC_W)
    return torch.sum(torch.square(torso_projected_gravity_b[:, :2]), dim=1)


def follow_command_vel_z(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reward tracking of linear velocity commands (z axis) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    command_term = env.command_manager.get_term(command_name)
    command_vel_z_w = command_term.command[..., 0]

    asset = env.scene["robot"]
    pelvis_vel_z_w = asset.data.root_lin_vel_w[:, 2]
    return torch.square((command_vel_z_w - pelvis_vel_z_w) / command_term.cfg.max_velocity)

def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # checking if contacts exist
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # retrieve velocity of feet
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    feet_links = ["left_ankle_roll_link", "right_ankle_roll_link"]
    feet_ids = asset.find_bodies(feet_links)[0]
    body_vel = asset.data.body_lin_vel_w[:, feet_ids, :2]

    # compute the reward as: velocity norm if there is a contact, otherwise zero.
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def standing_still_four_contact_points_v2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    left_foot_sensor_cfg: SceneEntityCfg,
    right_foot_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Discrete version of standing_still_four_contact_points which is binary."""

    command: mdp.EETrackCommand = env.command_manager.get_term(command_name)

    asset: Articulation = env.scene[asset_cfg.name]
    left_foot_sensor: ContactSensorExtra = env.scene.sensors[left_foot_sensor_cfg.name]
    right_foot_sensor: ContactSensorExtra = env.scene.sensors[right_foot_sensor_cfg.name]

    left_forces, left_points, left_masks = zmp.process_contact_data(
        left_foot_sensor.data.c_force,
        left_foot_sensor.data.c_normal,
        left_foot_sensor.data.c_point,
        left_foot_sensor.data.c_idx,
        left_foot_sensor.data.c_num,
    )

    right_forces, right_points, right_masks = zmp.process_contact_data(
        right_foot_sensor.data.c_force,
        right_foot_sensor.data.c_normal,
        right_foot_sensor.data.c_point,
        right_foot_sensor.data.c_idx,
        right_foot_sensor.data.c_num,
    )

    left_num_contacts = left_masks.sum(dim=-1)
    right_num_contacts = right_masks.sum(dim=-1)
    total_num_contacts = left_num_contacts + right_num_contacts
    total_num_sensors = 8

    rew = total_num_contacts / total_num_sensors

    # zero reward for moving envs
    moving_env_ids = (~command.is_standing_env).nonzero(as_tuple=False).flatten()
    rew[moving_env_ids] = 0.0
    return rew

def zmp_supp_dist_v2(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float,
    asset_cfg: SceneEntityCfg,
    left_foot_sensor_cfg: SceneEntityCfg,
    right_foot_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    # """
    # Computes the distance/margin between the support polygon and the zmp
    # """
    command: mdp.EETrackCommand = env.command_manager.get_term(command_name)

    asset: Articulation = env.scene[asset_cfg.name]

    zmp.calculate_zmp_pos_w(env, asset_cfg, left_foot_sensor_cfg, right_foot_sensor_cfg)

    # NOTE: hull_points: convex hull of left and right feet contact points.
    #       hull_idx: indices of the convex hull points.
    signed_zmp_dist = zmp.hull_point_signed_dist(
        asset.data.hull_points, asset.data.hull_idx, asset.data.zmp_pos_w[..., :2]
    )
    rew = torch.exp(sigma * -torch.clip(signed_zmp_dist, max=0)) - 1

    # Zero-mask reward for the moving command envs
    moving_env_ids = (~command.is_standing_env).nonzero(as_tuple=False).flatten()
    rew[moving_env_ids] = 0.0

    return rew

def ankle_parallel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Reward by the z_pos variance of ankle keypoints.
    Following HoST paper (https://arxiv.org/pdf/2502.08378)
    """

    def construct_keypoints(pos, quat):
        # keypoints in body frame
        kpts_b = (
            torch.tensor([[-0.05, 0.025, -0.03], [-0.05, -0.025, -0.03], [0.12, 0.03, -0.03], [0.12, -0.03, -0.03]])
            .to(pos.device)
            .repeat(pos.shape[0], 1, 1)
        )

        kpts_w = math_utils.transform_points(kpts_b, pos, quat)

        return kpts_w

    asset = env.scene[asset_cfg.name]
    feet_links = ["left_ankle_roll_link", "right_ankle_roll_link"]
    feet_ids = asset.find_bodies(feet_links)[0]

    feet_pos = asset.data.body_pos_w[:, feet_ids]
    feet_quat = asset.data.body_quat_w[:, feet_ids]

    left_feet_kpts = construct_keypoints(feet_pos[:, 0], feet_quat[:, 0])
    right_feet_kpts = construct_keypoints(feet_pos[:, 1], feet_quat[:, 1])

    z_var = left_feet_kpts[..., 2].var(dim=1) + right_feet_kpts[..., 2].var(dim=1)

    return (z_var < 0.05).float()