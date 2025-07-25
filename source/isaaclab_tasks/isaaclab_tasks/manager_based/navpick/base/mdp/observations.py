# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Optional, List

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def partial_joint_pos_rel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: Optional[List[str]] = None
) -> torch.Tensor:
    """The target joint positions of the asset w.r.t. the default joint positions.

    Args:
        env: Manager-based environment
        asset_cfg: Configuration for the scene entity
        joint_names: List of joint names to get the positions for. If None, all joints are used.

    Returns:
        Tensor of the target joint positions relative to the default joint positions.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if joint_names is not None:
        name_to_index = {name: idx for idx, name in enumerate(asset.joint_names)}
        try:
            joint_ids = [name_to_index[name] for name in joint_names]
        except KeyError as e:
            raise ValueError(f"joint name {e} not found in asset.joint_names") from e
    else:
        joint_ids = asset_cfg.joint_ids

    return asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]


def partial_joint_vel_rel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: Optional[List[str]] = None
) -> torch.Tensor:
    """The target joint velocities of the asset w.r.t. the default joint velocities.

    Args:
        env: Manager-based environment
        asset_cfg: Configuration for the scene entity
        joint_names: List of joint names to get the velocities for. If None, all joints are used.

    Returns:
        Tensor of the target joint velocities relative to the default joint velocities.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if joint_names is not None:
        name_to_index = {name: idx for idx, name in enumerate(asset.joint_names)}
        try:
            joint_ids = [name_to_index[name] for name in joint_names]
        except KeyError as e:
            raise ValueError(f"joint name {e} not found in asset.joint_names") from e
    else:
        joint_ids = asset_cfg.joint_ids

    return asset.data.joint_vel[:, joint_ids] - asset.data.default_joint_vel[:, joint_ids]