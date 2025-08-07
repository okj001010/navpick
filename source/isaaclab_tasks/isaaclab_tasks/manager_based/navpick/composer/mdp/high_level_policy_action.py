# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .pre_trained_policy_action import PreTrainedPolicyAction, PreTrainedPolicyActionCfg
from .. import upper_joint_names, lower_joint_names

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class HighLevelPolicyAction(ActionTerm):
    r"""High-level policy action term.

    This action term infers a high-level policy and applies the corresponding low-level actions to the robot.
    The raw actions correspond to the commands for the high-level policy.
    
    raw_actions = [ upper_logits | lower_logits | upper_command_1 | ... | lower_command_1 | ... ]

    """

    cfg: HighLevelPolicyActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: HighLevelPolicyActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        
        # upper and lower joint configurations
        self.upper_logits_dim = len(self.cfg.upper_actions)
        self.lower_logits_dim = len(self.cfg.lower_actions)
        self.upper_dim = len(upper_joint_names)
        self.lower_dim = len(lower_joint_names)
        self.upper_joint_ids, _ = self.robot.find_joints(upper_joint_names)
        self.lower_joint_ids, _ = self.robot.find_joints(lower_joint_names)
    
        # prepare the pre-trained policy action terms
        self._prepare_terms(cfg)
        
        # prepare the buffers
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.upper_weights = torch.zeros(self.num_envs, self.upper_logits_dim, device=self.device)
        self.lower_weights = torch.zeros(self.num_envs, self.lower_logits_dim, device=self.device)


    """
    Helper functions.
    """
    def _prepare_terms(self, cfg: HighLevelPolicyActionCfg):
        # create buffers to parse and store terms
        self._upper_term_names: list[str] = list()
        self._upper_terms: dict[str, PreTrainedPolicyAction] = dict()
        self._lower_term_names: list[str] = list()
        self._lower_terms: dict[str, PreTrainedPolicyAction] = dict()
        
        offset = self.upper_logits_dim + self.lower_logits_dim

        # parse upper action terms from the config
        for upper_term_name, upper_term_cfg in cfg.upper_actions.items():
            upper_term_cfg.asset_name = cfg.asset_name
            upper_term_cfg.low_level_decimation = cfg.low_level_decimation
            upper_term_command = upper_term_cfg.low_level_env_cfg.observations.policy.__dict__[upper_term_cfg.command_name]
            upper_term_command.func = (lambda o: lambda dummy_env: self._raw_actions[:, o:o + upper_term_command.command_dim])(offset)
            upper_term_command.params = dict()
            upper_term = upper_term_cfg.class_type(upper_term_cfg, self._env)
            offset += upper_term.command_dim
            self._upper_term_names.append(upper_term_name)
            self._upper_terms[upper_term_name] = upper_term
        
        # parse lower action terms from the config
        for lower_term_name, lower_term_cfg in cfg.lower_actions.items():
            lower_term_cfg.asset_name = cfg.asset_name
            lower_term_cfg.low_level_decimation = cfg.low_level_decimation
            lower_term_command = lower_term_cfg.low_level_env_cfg.observations.policy.__dict__[lower_term_cfg.command_name]
            lower_term_command.func = (lambda o: lambda dummy_env: self._raw_actions[:, o:o + lower_term_command.command_dim])(offset)
            lower_term_command.params = dict()
            lower_term = lower_term_cfg.class_type(lower_term_cfg, self._env)
            self._lower_term_names.append(lower_term_name)
            self._lower_terms[lower_term_name] = lower_term


    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self.upper_logits_dim + self.lower_logits_dim + self.command_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    @property
    def command_dim(self) -> int:
        """Total dimension of the commands."""
        return sum(term.command_dim for term in self._upper_terms.values()) + \
                sum(term.command_dim for term in self._lower_terms.values())

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions

        offset = 0
        upper_logits = actions[:, offset:offset+self.upper_logits_dim]
        self.upper_weights = F.softmax(upper_logits / self.cfg.temperature, dim=-1)
        self._processed_actions[:, :self.upper_logits_dim] = upper_logits
        
        offset += self.upper_logits_dim
        lower_logits = actions[:, offset:offset+self.lower_logits_dim]
        self.lower_weights = F.softmax(lower_logits / self.cfg.temperature, dim=-1)
        self._processed_actions[:, offset:offset+self.lower_logits_dim] = lower_logits


    def apply_actions(self):
        # Apply weighted sum of policy outputs
        upper_action = torch.zeros(self.num_envs, self.upper_dim, device=self.device)
        for i, term in enumerate(self._upper_terms.values()):
            upper_action += self.upper_weights[:, i:i+1] * term.actions

        lower_action = torch.zeros(self.num_envs, self.lower_dim, device=self.device)
        for i, term in enumerate(self._lower_terms.values()):
            lower_action += self.lower_weights[:, i:i+1] * term.actions

        self.robot.set_joint_position_target(upper_action, joint_ids=self.upper_joint_ids)
        self.robot.set_joint_position_target(lower_action, joint_ids=self.lower_joint_ids)


@configclass
class HighLevelPolicyActionCfg(ActionTermCfg):
    """Configuration for high-level policy action term.

    See :class:`HighLevelPolicyAction` for more details.
    """

    class_type: type[ActionTerm] = HighLevelPolicyAction
    """ Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    upper_actions: dict[str, PreTrainedPolicyActionCfg] = MISSING
    """List of upper-body actions to be applied."""
    lower_actions: dict[str, PreTrainedPolicyActionCfg] = MISSING
    """List of lower-body actions to be applied."""
    temperature: float = 1.0
    """Temperature for the softmax distribution over the actions."""