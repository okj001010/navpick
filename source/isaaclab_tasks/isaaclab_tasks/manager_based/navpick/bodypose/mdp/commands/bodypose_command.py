# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators that does nothing."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .bodypose_command_cfg import BodyPoseCommandCfg


class BodyPoseCommand(CommandTerm):
    """Command generator for body pose commands."""

    cfg: BodyPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: BodyPoseCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        self.env = env
        self.robot: Articulation = env.scene[cfg.asset_name]

        # initialize the command sampler
        self.init_command_sampler(self.cfg.ranges)

        # find the joint indices
        self.left_hip_pitch = self.robot.find_joints(cfg.left_hip_pitch_joint)[0][0]
        self.right_hip_pitch = self.robot.find_joints(cfg.right_hip_pitch_joint)[0][0]

        # create buffers to store the command
        self.goal_base_height = torch.zeros((self.env.num_envs,), dtype=torch.float32, device=self.env.device)
        self.goal_hip_pitch = torch.zeros((self.env.num_envs,), dtype=torch.float32, device=self.env.device)
        
        # metrics
        self.metrics["error_base_height"] = torch.zeros((env.num_envs,), device=env.device)
        self.metrics["error_hip_pitch"] = torch.zeros((env.num_envs,), device=env.device)
    
    """
    Helper functions
    """
    def init_command_sampler(self, ranges: BodyPoseCommandCfg.Ranges):
        self.base_height_sampler = torch.distributions.Uniform(
            low=torch.tensor(ranges.base_height[0], device=self.env.device),
            high=torch.tensor(ranges.base_height[1], device=self.env.device),
        )
        self.hip_pitch_sampler = torch.distributions.Uniform(
            low=torch.tensor(ranges.hip_pitch[0], device=self.env.device),
            high=torch.tensor(ranges.hip_pitch[1], device=self.env.device),
        )


    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired body pose (base height and hip pitch) command. Shape is (num_envs, 2)."""
        return torch.stack((self.goal_base_height, self.goal_hip_pitch), dim=-1)


    """
    Operations
    """
    
    # FIXME(OKJ): maybe something metrics related things can be wrong
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset the metrics"""
        extras = {}
        for name, metric in self.metrics.items():
            extras[name] = metric.mean().item()
            metric.zero_()
        return extras


    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        self.metrics["error_base_height"] += torch.abs(self.robot.data.root_state_w[:, 2] - self.goal_base_height)
        left_hip_pitch_error = torch.abs(self.robot.data.joint_pos[:, self.left_hip_pitch] - self.goal_hip_pitch)
        right_hip_pitch_error = torch.abs(self.robot.data.joint_pos[:, self.right_hip_pitch] - self.goal_hip_pitch)
        self.metrics["error_hip_pitch"] += left_hip_pitch_error + right_hip_pitch_error

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the command for the given environment IDs."""
        self.goal_base_height[env_ids] = self.base_height_sampler.sample((len(env_ids), 1))
        self.goal_hip_pitch[env_ids] = self.hip_pitch_sampler.sample((len(env_ids), 1))

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Set debug visualization implementation."""
        if debug_vis:
            if not hasattr(self, "goal_base_visualizer"):
                self.goal_base_visualizer = VisualizationMarkers(self.cfg.goal_base_visualizer_cfg)
                self.current_base_visualizer = VisualizationMarkers(self.cfg.current_base_visualizer_cfg)
                self.goal_left_hip_pitch_visualizer = VisualizationMarkers(self.cfg.goal_left_hip_pitch_visualizer_cfg)
                self.goal_right_hip_pitch_visualizer = VisualizationMarkers(self.cfg.goal_right_hip_pitch_visualizer_cfg)
                self.current_left_hip_pitch_visualizer = VisualizationMarkers(self.cfg.current_left_hip_pitch_visualizer_cfg)
                self.current_right_hip_pitch_visualizer = VisualizationMarkers(self.cfg.current_right_hip_pitch_visualizer_cfg)
            # set their visibility to true
            self.goal_base_visualizer.set_visibility(True)
            self.current_base_visualizer.set_visibility(True)
            self.goal_left_hip_pitch_visualizer.set_visibility(True)
            self.goal_right_hip_pitch_visualizer.set_visibility(True)
            self.current_left_hip_pitch_visualizer.set_visibility(True)
            self.current_right_hip_pitch_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_base_visualizer"):
                self.goal_base_visualizer.set_visibility(False)
                self.current_base_visualizer.set_visibility(False)
                self.goal_left_hip_pitch_visualizer.set_visibility(False)
                self.goal_right_hip_pitch_visualizer.set_visibility(False)
                self.current_left_hip_pitch_visualizer.set_visibility(False)
                self.current_right_hip_pitch_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Debug visualization callback."""
        if not self.robot.is_initialized:
            return
        
        # visualize the goal base height
        goal_base_pos = self.robot.data.root_state_w[:, :3].clone()
        goal_base_pos[:, 2] = self.goal_base_height
        self.goal_base_visualizer.visualize(goal_base_pos)
        
        # visualize the current base height
        current_base_pos = self.robot.data.root_state_w[:, :3].clone()
        self.current_base_visualizer.visualize(current_base_pos)
        
        # visualize the goal and current hip pitch angles with scaled arrows
        arrow_euler = torch.tensor([0.0, -math.pi / 2, 0.0], device=self.env.device).expand(self.env.num_envs, -1)
        arrow_quat = math_utils.quat_from_euler_xyz(arrow_euler[:, 0], arrow_euler[:, 1], arrow_euler[:, 2])
        left_arrow_pos = self.robot.data.root_state_w[:, :3].clone()
        left_arrow_pos[:, 2] = 0
        left_arrow_pos[:, 0] -= 0.1
        right_arrow_pos = self.robot.data.root_state_w[:, :3].clone()
        right_arrow_pos[:, 2] = 0
        right_arrow_pos[:, 0] += 0.1
        goal_left_hip_arrow_scale = self._resolve_arrow_scale(self.goal_hip_pitch)
        goal_right_hip_arrow_scale = self._resolve_arrow_scale(self.goal_hip_pitch)
        current_left_hip_arrow_scale = self._resolve_arrow_scale(self.robot.data.joint_pos[:, self.left_hip_pitch])
        current_right_hip_arrow_scale = self._resolve_arrow_scale(self.robot.data.joint_pos[:, self.right_hip_pitch])
        self.goal_left_hip_pitch_visualizer.visualize(left_arrow_pos, arrow_quat, goal_left_hip_arrow_scale)
        self.goal_right_hip_pitch_visualizer.visualize(right_arrow_pos, arrow_quat, goal_right_hip_arrow_scale)
        self.current_left_hip_pitch_visualizer.visualize(left_arrow_pos, arrow_quat, current_left_hip_arrow_scale)
        self.current_right_hip_pitch_visualizer.visualize(right_arrow_pos, arrow_quat, current_right_hip_arrow_scale)
    
    def _resolve_arrow_scale(self, angles: torch.Tensor) -> torch.Tensor:
        """Resolve the arrow scale based on the hip pitch angle (magnitude)."""
        #TODO(OKJ): fix the scale of the arrow
        return torch.abs(angles) * 0.1
