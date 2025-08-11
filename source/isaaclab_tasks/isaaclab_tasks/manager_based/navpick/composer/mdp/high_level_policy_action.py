# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.markers import VisualizationMarkers, RED_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from .pre_trained_policy_action import PreTrainedPolicyAction, PreTrainedPolicyActionCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class HighLevelPolicyAction(ActionTerm):
    r"""High-level policy action term.

    This action term infers a high-level policy and applies the corresponding low-level actions to the robot.
    The raw actions correspond to the commands for the high-level policy.
    
    raw_actions = [ logits (1, 2, ...) | command_1 | command_2 | ... ]

    """

    cfg: HighLevelPolicyActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: HighLevelPolicyActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.logits_dim = len(cfg.low_level_actions)
        
        # prepare the buffers
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.low_level_weights = torch.zeros(self.num_envs, self.logits_dim, device=self.device)
        
        # prepare the pre-trained policy action terms
        self._prepare_terms(cfg)
        
        if cfg.test_mode:
            print("HighLevelPolicyAction: Using argmax actions only.")


    """
    Helper functions.
    """
    def _prepare_terms(self, cfg: HighLevelPolicyActionCfg):
        # create buffers to parse and store terms
        self._low_level_term_names: list[str] = list()
        self._low_level_terms: dict[str, PreTrainedPolicyAction] = dict()
        
        offset = self.logits_dim

        # parse low level action terms from the config
        for low_level_term_name, low_level_term_cfg in cfg.low_level_actions.items():
            
            # low level command
            low_level_obs_cfg = low_level_term_cfg.low_level_env_cfg.observations.__dict__.get(low_level_term_cfg.observation_name)
            low_level_command = low_level_obs_cfg.__dict__.get(low_level_term_cfg.command_name)

            # high-level policy action -> low-level policy observation
            low_level_command.func = (
                lambda dummy_env, o=offset, d=low_level_term_cfg.command_dim:
                    self._raw_actions[:, o:o+d]
            )
            low_level_command.params = {}
            offset += low_level_term_cfg.command_dim

            # store the low level term
            low_level_term = low_level_term_cfg.class_type(low_level_term_cfg, self._env)
            self._low_level_term_names.append(low_level_term_name)
            self._low_level_terms[low_level_term_name] = low_level_term


    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self.logits_dim + self.command_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    @property
    def command_dim(self) -> int:
        """Total dimension of the commands."""
        return sum(term.command_dim for term in self.cfg.low_level_actions.values())

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions

        logits = actions[:, :self.logits_dim]
        if self.cfg.test_mode:
            self.low_level_weights = F.one_hot(torch.argmax(logits, dim=-1), num_classes=self.logits_dim).float()
        else:
            self.low_level_weights = F.softmax(logits / self.cfg.temperature, dim=-1)
        self._processed_actions[:, :self.logits_dim] = logits


    def apply_actions(self):
        # Apply weighted sum of policy outputs
        mixed_low_level_action = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        for i, term in enumerate(self._low_level_terms.values()):
            low_level_action = term.actions
            mixed_low_level_action += self.low_level_weights[:, i:i+1] * low_level_action

        self.robot.set_joint_position_target(mixed_low_level_action)
    
    
    """
    Debug visualization.
    """
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization implementation."""
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "low_level_weights_visualizers"):
                self.low_level_weights_visualizers = []
                for i, term_name in enumerate(self.cfg.low_level_actions.keys()):
                    arrow_cfg = RED_ARROW_X_MARKER_CFG.copy()
                    arrow_cfg.prim_path = f"/Visuals/Actions/{term_name}/weights"
                    arrow_cfg.markers["arrow"].scale = (0.1, 0.1, 0.1)
                    arrow_cfg.markers["arrow"].visual_material.diffuse_color = (float(i % 3 == 0), float(i % 3 == 1), float(i % 3 == 2))
                    self.low_level_weights_visualizers.append(VisualizationMarkers(arrow_cfg))
            # set their visibility to true
            for visualizer in self.low_level_weights_visualizers:
                visualizer.set_visibility(True)
        else:
            if hasattr(self, "low_level_weights_visualizer"):
                for visualizer in self.low_level_weights_visualizers:
                    visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Debug visualization callback."""
        if not self.robot.is_initialized:
            return

        side_direction = torch.stack((-torch.sin(self.robot.data.heading_w), torch.cos(self.robot.data.heading_w), torch.zeros(self.num_envs, device=self.device)), dim=-1)
        head_pos = self.robot.data.root_state_w[:, :3].clone()
        head_pos[:, 2] += 0.8
        arrow_quat = math_utils.quat_from_euler_xyz(
            torch.zeros(self.num_envs, device=self.device),
            -torch.ones(self.num_envs, device=self.device) * math.pi / 2,
            torch.zeros(self.num_envs, device=self.device),
        )
        
        center_index = (len(self.low_level_weights_visualizers) - 1) / 2
        for i, visualizer in enumerate(self.low_level_weights_visualizers):
            # visualize the low-level action weights
            offset_index = i - center_index
            arrow_pos = head_pos + 0.2 * side_direction * offset_index
            arrow_scale = torch.ones((self.num_envs, 3), device=self.device)
            arrow_scale[:, 0] *= 5 * self.low_level_weights[:, i] + 0.01  # add a small offset to avoid zero length arrows
            visualizer.visualize(arrow_pos, arrow_quat, arrow_scale)



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
    low_level_actions: dict[str, PreTrainedPolicyActionCfg] = MISSING
    """List of low-level actions to be applied."""
    temperature: float = 1.0
    """Temperature for the softmax distribution over the actions."""
    test_mode: bool = False
    """If True, only the argmax of the logits is used instead of linear combination"""
    debug_vis: bool = True
    """Whether to visualize debug information. Defaults to False."""