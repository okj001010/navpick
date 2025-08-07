# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators that does nothing."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .sit_command_cfg import SitCommandCfg


class SitCommand(CommandTerm):
    """Command generator for sit commands."""

    cfg: SitCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: SitCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        self.env = env
        self.robot: Articulation = env.scene[cfg.asset_name]

        # create buffers to store the command
        self.pelvis_height_w = torch.zeros((self.env.num_envs,), dtype=torch.float32, device=self.env.device)
        self.pelvis_lin_vel_z_w = torch.zeros((self.env.num_envs,), dtype=torch.float32, device=self.env.device)
        
        # metrics
        self.metrics["height_error"] = torch.zeros((env.num_envs,), device=env.device)
        self.metrics["velocity_error"] = torch.zeros((env.num_envs,), device=env.device)


    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired sit command (pelvis velocity and pelvis height). Shape is (num_envs, 2)."""
        return torch.stack((self.pelvis_lin_vel_z_w, self.pelvis_height_w), dim=-1)


    """
    Operations
    """
    
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset the metrics"""
        extras = {}
        for name, metric in self.metrics.items():
            extras[name] = torch.mean(metric[env_ids])
            metric.zero_()
        return extras


    """
    Implementation specific functions.
    """
    def _update_target_height_and_lin_vel_z_w(self):
        """Update the target height based on the current root position."""
        height_diff = self.pelvis_height_w - self.robot.data.root_pos_w[:, 2]

        option = self.cfg.velocity_command_option
        if option == "sqrt":
            self.pelvis_lin_vel_z_w[:] = torch.clamp(
                torch.sign(height_diff)
                * self.cfg.max_velocity
                * torch.sqrt(torch.abs(height_diff / self.cfg.slow_bound)),
                max=self.cfg.max_velocity,
                min=-self.cfg.max_velocity,
            )
        elif option == "linear":
            self.pelvis_lin_vel_z_w[:] = torch.clamp(
                self.cfg.max_velocity * (height_diff / self.cfg.slow_bound),
                max=self.cfg.max_velocity,
                min=-self.cfg.max_velocity,
            )
        else:
            print("Velocity command option is only available between sqrt and linear.")
            raise NotImplementedError

    def _update_metrics(self):
        self.metrics["height_error"] = torch.abs(self.robot.data.root_state_w[:, 2] - self.pelvis_height_w)
        self.metrics["velocity_error"] = torch.abs(self.pelvis_lin_vel_z_w - self.robot.data.root_vel_w[:, 2])

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the command for the given environment IDs."""
        r = torch.empty(len(env_ids), device=self.device)
        self.pelvis_height_w[env_ids] = r.uniform_(*self.cfg.ranges.height)
        self._update_target_height_and_lin_vel_z_w()

    def _update_command(self):
        """Re-target the height & velocity command based on the current root position."""
        self._update_target_height_and_lin_vel_z_w()

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Set debug visualization implementation."""
        if debug_vis:
            if not hasattr(self, "goal_height_visualizer"):
                self.goal_height_visualizer = VisualizationMarkers(self.cfg.goal_height_visualizer_cfg)
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_height_visualizer.set_visibility(True)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_height_visualizer.set_visibility(False)
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Debug visualization callback."""
        if not self.robot.is_initialized:
            return
        
        # visualize the goal height
        height_command_pos = self.robot.data.root_pos_w.clone()
        height_command_pos[:, 2] = self.pelvis_height_w
        height_command_ori = torch.zeros_like(self.robot.data.root_quat_w)
        height_command_ori[:, 0] = 1.0
        self.goal_height_visualizer.visualize(translations=height_command_pos, orientations=height_command_ori)
        
        # visualize the goal and current velocities
        velocity_command_pos = self.robot.data.root_pos_w.clone()
        velocity_command_pos[:, 2] += 0.7  # offset the height for better visualization        
        goal_vel_arrow_scale, goal_vel_arrow_quat = self._resolve_z_velocity_to_arrow(self.pelvis_lin_vel_z_w)
        current_vel_arrow_scale, current_vel_arrow_quat = self._resolve_z_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, 2])

        self.goal_vel_visualizer.visualize(velocity_command_pos, goal_vel_arrow_scale, goal_vel_arrow_quat)
        self.current_vel_visualizer.visualize(velocity_command_pos, current_vel_arrow_quat, current_vel_arrow_scale)

    def _resolve_z_velocity_to_arrow(self, z_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # arrow scale
        arrow_scale = torch.ones((self.env.num_envs, 3), device=self.device)
        arrow_scale[:, 0] *= torch.abs(z_velocity) * 30.0

        # arrow quaternion
        arrow_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(z_velocity),
            -torch.sign(z_velocity) * math.pi / 2,
            torch.zeros_like(z_velocity),
        )

        return arrow_scale, arrow_quat