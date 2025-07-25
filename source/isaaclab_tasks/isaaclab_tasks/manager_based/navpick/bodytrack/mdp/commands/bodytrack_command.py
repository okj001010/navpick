# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .eetrack_command_cfg import BodyTrackCommandCfg


class BodyTrackCommand(CommandTerm):
    """Command generator for generating root xy position command uniformly.

    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space.

    The position command are generated in the base frame of the robot, and not the
    simulation world frame. This means that users need to handle the transformation from the
    base frame to the simulation world frame themselves.
    """

    cfg: BodyTrackCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: BodyTrackCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        
        # intial robot state
        self.init_root_xy_pos_w = torch.zeros(self.num_envs, 2, device=self.device)

        # create command buffers
        self.goal_command_b = torch.zeros(self.num_envs, 2, device=self.device)
        self.goal_command_w = torch.zeros(self.num_envs, 2, device=self.device)
        
        # initialize metrics
        self._initialize_metrics()
        
        # goal achievement tracking for curriculum
        self.goal_achieved_ratio_ema = 0.0

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired root xy position command. Shape is (num_envs, 2) in (x, y) format."""
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        
        # calculate position difference in world frame
        diff_pos = torch.zeros(self.num_envs, 3, device=self.device)
        diff_pos[:, :2] = self.goal_command_w - root_pos_w[:, :2]
        
        # transform to base frame
        self.goal_command_b = math_utils.quat_apply_inverse(math_utils.yaw_quat(root_quat_w), diff_pos)[:, :2]
        return self.goal_command_b
    
    """
    Operations
    """
    
    def _initialize_metrics(self):
        """Define and initialize metrics"""
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_achieved_ratio"] = torch.zeros(1, device=self.device)
    
    def _calculate_goal_achievements(self, env_ids: Sequence[int]) -> list[float]:
        """Calculate goal achievements for given env_ids."""
        goal_achievements = []
        for env_id in env_ids:
            distance_to_goal = self.metrics["position_error"][env_id].item()
            is_achieved = distance_to_goal < self.cfg.goal_achievement_threshold
            goal_achievements.append(float(is_achieved))
        
        return goal_achievements
    
    def _update_goal_achieved_ratio_metric(self, env_ids: Sequence[int]):
        """Update EMA of goal achievement ratio."""
        if len(env_ids) == 0:
            return
        
        # calculate goal achievements for resetting environments
        goal_achievements = self._calculate_goal_achievements(env_ids)
            
        current_success_rate = sum(goal_achievements) / len(goal_achievements)
        
        if self.goal_achieved_ratio_ema == 0.0:  # First update
            self.goal_achieved_ratio_ema = current_success_rate
        else:
            self.goal_achieved_ratio_ema = (
                self.cfg.ema_alpha * current_success_rate + 
                (1 - self.cfg.ema_alpha) * self.goal_achieved_ratio_ema
            )
        
        self.metrics["goal_achieved_ratio"][0] = self.goal_achieved_ratio_ema
        
    def _prepare_reset_extras(self, env_ids: Sequence[int]) -> dict[str, float]:
        """Log metrics at reset."""
        extras = {}
        extras["position_error"] = torch.mean(self.metrics["position_error"][env_ids]).item()
        extras["goal_achieved_ratio"] = self.metrics["goal_achieved_ratio"][0].item()
        return extras
    
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # Resolve the environment IDs
        if env_ids is None:
            env_ids = slice(env_ids)
        
        # update every metrics last time before reset
        self._update_metrics()
        self._update_goal_achieved_ratio_metric(env_ids)
        
        # resample the command
        self.command_counter[env_ids] = 0
        self._resample(env_ids)
        
        # save initial root xy position in world frame
        self.init_root_xy_pos_w[env_ids] = self.robot.data.root_state_w[env_ids, :2]
        
        return self._prepare_reset_extras(env_ids)
    

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics every step."""
        current_root_pos_w = self.robot.data.root_pos_w[:, :2]
        pos_error = self.goal_command_w - current_root_pos_w
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        r = torch.empty(len(env_ids), device=self.device)
        self.goal_command_w[env_ids, :] = self.init_root_xy_pos_w[env_ids, :]
        self.goal_command_w[env_ids, 0] += r.uniform_(*self.cfg.ranges.root_x)
        self.goal_command_w[env_ids, 1] += r.uniform_(*self.cfg.ranges.root_y)

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Set debug visualization implementation."""
        if debug_vis:
            self._create_visualizers()
            self._set_visualizers_visibility(True)
        else:
            self._set_visualizers_visibility(False)

    def _create_visualizers(self) -> None:
        """Create visualization markers if they don't exist."""
        if not hasattr(self, "goal_root_pos_visualizer"):
            self.goal_root_pos_visualizer = VisualizationMarkers(
                self.cfg.goal_root_pos_visualizer_cfg
            )
            self.current_root_pos_visualizer = VisualizationMarkers(
                self.cfg.current_root_pos_visualizer_cfg
            )

    def _set_visualizers_visibility(self, visible: bool) -> None:
        """Set visibility of visualization markers."""
        if hasattr(self, "goal_root_pos_visualizer"):
            self.goal_root_pos_visualizer.set_visibility(visible)
            self.current_root_pos_visualizer.set_visibility(visible)
    
    def _debug_vis_callback(self, event):
        """Debug visualization callback."""
        if not self.robot.is_initialized:
            return
        
        # update goal position visualization markers
        vis_goal_root_pos_w = torch.zeros_like(self.robot.data.root_pos_w)
        vis_goal_root_pos_w[:, :2] = self.goal_command_w[:, :2]
        vis_goal_root_pos_w[:, 2] = 0.1
        self.goal_root_pos_visualizer.visualize(vis_goal_root_pos_w)
        
        # update current position visualization with arrow pointing to goal
        arrow_start_pos, diff_arrow_quat, diff_arrow_scale = self._calculate_arrow_parameters(
            self.robot.data.root_pos_w, self.goal_command_w
        )
        self.current_root_pos_visualizer.visualize(arrow_start_pos, diff_arrow_quat, diff_arrow_scale)

    """
    Internal helpers.
    """

    def _calculate_arrow_parameters(
        self, current_pos_w: torch.Tensor, goal_pos_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Converts the position difference to arrow starting from current position pointing to goal."""
        # Calculate position difference in world frame
        pos_diff_w = goal_pos_w[:, :2] - current_pos_w[:, :2]
        
        # obtain default scale of the marker
        default_scale = self.current_root_pos_visualizer.cfg.markers["arrow"].scale
        
        # Calculate arrow length based on distance to goal
        arrow_length = torch.linalg.norm(pos_diff_w, dim=1)
        
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(current_pos_w.shape[0], 1)
        arrow_scale[:, 0] = arrow_length
        
        # arrow-direction
        heading_angle = torch.atan2(pos_diff_w[:, 1], pos_diff_w[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        
        # Calculate arrow start position
        direction_unit = pos_diff_w / (torch.linalg.norm(pos_diff_w, dim=1, keepdim=True) + 1e-8)
        arrow_offset = direction_unit * (arrow_length.unsqueeze(1) * default_scale[0] / 2.0)
        
        arrow_start_pos = torch.zeros_like(current_pos_w)
        arrow_start_pos[:, :2] = current_pos_w[:, :2] + arrow_offset
        arrow_start_pos[:, 2] = 0.1

        return arrow_start_pos, arrow_quat, arrow_scale
