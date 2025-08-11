# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING, cast

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ObservationManager
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file
from isaaclab.envs.mdp.actions import JointPositionAction

from ...base import base_joint_names

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class PreTrainedPolicyAction:
    r"""Pre-trained policy action term. (strtictly speaking, this is not an action term)

    This action term infers a pre-trained policy and applies the corresponding low-level actions to the robot.
    The raw actions correspond to the commands for the pre-trained policy.

    """

    cfg: PreTrainedPolicyActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: PreTrainedPolicyActionCfg, env: ManagerBasedEnv) -> None:
        # store the configuration and environment
        self.cfg = cfg
        self.env = env
        self.robot: Articulation = env.scene[cfg.asset_name]
        
        # load policy
        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        file_bytes = read_file(cfg.policy_path)
        self.policy = torch.jit.load(file_bytes).to(env.device).eval()

        # prepare low level actions
        low_level_action_cfg = cfg.low_level_env_cfg.actions.__dict__.get(cfg.action_name)
        self._low_level_action_term: ActionTerm = low_level_action_cfg.class_type(low_level_action_cfg, env)
        self._low_level_actions = torch.zeros(self.num_envs, self._low_level_action_term.action_dim, device=self.device)
        
        # prepare low level action joint ids
        low_level_action_joint_names = low_level_action_cfg.joint_names
        self._low_level_action_joint_ids, _ = self.robot.find_joints(low_level_action_joint_names)
        self._non_low_level_action_joint_ids = [joint_id for joint_id in range(self.robot.num_joints) if joint_id not in self._low_level_action_joint_ids]

        def last_action():
            # reset the low level actions if the episode was reset
            if hasattr(env, "episode_length_buf"):
                self._low_level_actions[self.env.episode_length_buf == 0, :] = 0
            return self._low_level_actions

        # remap some of the low level observations to internal observations
        cfg.low_level_env_cfg.observations.policy.actions.func = lambda dummy_env: last_action()
        cfg.low_level_env_cfg.observations.policy.actions.params = dict()

        # add the low level observations to the observation manager
        low_level_obs_cfg = cfg.low_level_env_cfg.observations.__dict__.get(cfg.observation_name)
        self._low_level_obs_manager = ObservationManager({"ll_policy": low_level_obs_cfg}, env)

        self._counter = 0

    """
    Properties.
    """
    
    @property
    def num_envs(self) -> int:
        return self.env.num_envs
    
    @property
    def device(self) -> str:
        return self.env.device
    
    @property
    def command_dim(self) -> int:
        return self.cfg.command_dim

    @property
    def actions(self) -> torch.Tensor:
        if self._counter % self.cfg.low_level_decimation == 0:
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")
            self._low_level_actions[:] = self.policy(low_level_obs)
            self._low_level_action_term.process_actions(self._low_level_actions)
            self._counter = 0
        self._counter += 1
            
        # low level full actions
        # - low level action joints -> set the low level actions
        # - non low level action joints -> keep the current joint positions
        low_level_full_actions = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        low_level_full_actions[:, self._low_level_action_joint_ids] = self._low_level_action_term.processed_actions
        low_level_full_actions[:, self._non_low_level_action_joint_ids] = self.robot.data.joint_pos[:, self._non_low_level_action_joint_ids]
        return low_level_full_actions


@configclass
class PreTrainedPolicyActionCfg:
    """Configuration for pre-trained policy action term.

    See :class:`PreTrainedPolicyAction` for more details.
    """

    class_type = PreTrainedPolicyAction
    """Class of the action term."""
    action_name: str = "joint_pos"
    """Name of the action term."""
    observation_name: str = "policy"
    """Name of the observation term."""
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    low_level_decimation: int = MISSING
    """Decimation factor for the low level action term."""
    low_level_env_cfg: ManagerBasedRLEnv = MISSING
    """Configuration for the low level environment."""
    policy_path: str = MISSING
    """Path to the low level policy (.pt files)."""
    command_name: str = MISSING
    """Name of the command to be generated by the high-level policy."""
    command_dim: int = MISSING
    """Dimension of the command to be generated by the high-level policy."""