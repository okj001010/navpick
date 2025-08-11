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
from isaaclab.managers import SceneEntityCfg, RewardTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    

def is_terminated_term(
    env: ManagerBasedRLEnv,
    term_names: list[str]
) -> torch.Tensor:
    """Penalize termination for specific terms that don't correspond to episodic timeouts."""
    reset_buf = torch.zeros(env.num_envs, device=env.device)
    for term in term_names:
        reset_buf += env.termination_manager.get_term(term)
    return (reset_buf * (~env.termination_manager.time_outs)).float()


def rel_joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    rel_torque = asset.data.applied_torque / asset.root_physx_view.get_dof_max_forces().clone().to(env.device)
    rew = torch.sum(torch.square(rel_torque[:, asset_cfg.joint_ids]), dim=1)
    return rew


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    energy = torch.clip(
        asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids],
        min=0,
    )
    return torch.sum(energy, dim=-1)


def action_acc_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the L2 norm of the action acceleration."""
    return torch.sum(
        torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action),
        dim=1,
    )


def default_joint_error(
    env: ManagerBasedRLEnv,
    joint_names: Optional[List[str]] = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
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