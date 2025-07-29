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
from typing import TYPE_CHECKING, Optional, List

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_acc_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the L2 norm of the action acceleration."""
    return torch.sum(
        torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action),
        dim=1,
    )

# def collision_penalty(
#     env: ManagerBasedRLEnv,
#     sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
#     force_threshold: float = 0.1,
# ) -> torch.Tensor:
#     """Penalize collisions with the environment."""
#     # FIXME(OKJ): remove ankle roll pitch and ground plane contact
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > force_threshold
#     return contacts


def default_joint_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: Optional[List[str]] = None
) -> torch.Tensor:
    """Penalize the joint position error from the default position."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    if joint_names is not None:
        name_to_index = {name: idx for idx, name in enumerate(asset.joint_names)}
        try:
            joint_ids = [name_to_index[name] for name in joint_names]
        except KeyError as e:
            raise ValueError(f"joint name {e} not found in asset.joint_names") from e
    else:
        joint_ids = asset_cfg.joint_ids

    return torch.sum(
        torch.exp(-2 * torch.square(asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids])),
        dim=1,
    )