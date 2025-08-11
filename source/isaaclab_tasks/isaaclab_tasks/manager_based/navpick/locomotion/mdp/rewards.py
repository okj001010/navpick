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
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def air_time_v4(
        env: ManagerBasedRLEnv, 
        command_name: str,
        threshold: float = 0.4,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
        ) -> torch.Tensor:


    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    rew = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]

    rew = torch.where(
        rew < threshold,
        rew,
        2 * threshold - rew
    )
    rew = torch.clamp(rew, min=0.)

    rew *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1

    return rew

def feet_contact_force(
    env: ManagerBasedRLEnv,
    # max_contact_force = 370.,
    max_contact_force = 400.,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
    ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation
    max_cforce = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]

    # Prevent penalty when initialized
    rew = torch.sum((max_cforce -  max_contact_force).clip(min=0.), dim=1) * \
        (env.episode_length_buf * env.step_dt >= 1.0).float()
    
    return rew

def feet_swing_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
    ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    contact = contact_time > 0.0

    asset: Articulation = env.scene[asset_cfg.name]
    pos_error = torch.square(asset.data.body_state_w[..., asset_cfg.body_ids, 2] - 0.08) * ~contact
    return torch.sum(pos_error, dim=1)