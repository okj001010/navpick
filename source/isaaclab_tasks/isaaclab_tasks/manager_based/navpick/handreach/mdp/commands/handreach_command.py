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
    from .handreach_command_cfg import HandReachCommandCfg


class HandReachCommand(CommandTerm):
    """Command generator for hand reach commands."""

    cfg: HandReachCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: HandReachCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        self.env = env
        self.robot: Articulation = env.scene[cfg.asset_name]
        
        # find the body indices
        self.target_hand_idx = self.robot.find_bodies(cfg.target_hand_name)[0][0]
        self.torso_idx = self.robot.find_bodies(cfg.torso_body_name)[0][0]
        
        # create buffers to store the command
        self.goal_hand_pose_b = torch.zeros((env.num_envs, 7), device=env.device)
        self.goal_hand_pose_w = torch.zeros((env.num_envs, 7), device=env.device)
        self.curr_hand_pose_b = torch.zeros((env.num_envs, 7), device=env.device)
        self.curr_hand_pose_w = torch.zeros((env.num_envs, 7), device=env.device)
        self.shoulder_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
        
        # metrics
        self.metrics["error_hand_pos"] = torch.zeros((env.num_envs,), device=env.device)
        self.metrics["error_hand_rot"] = torch.zeros((env.num_envs,), device=env.device)

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired hand pose command in the base frame. Shape is (num_envs, 6)."""
        return torch.cat((
            self.goal_hand_pose_b[:, :3],
            math_utils.axis_angle_from_quat(self.goal_hand_pose_b[:, 3:7])
        ), dim=-1)
    
    @property
    # return curr_hand_pose, goal_hand_pose in the base frame
    def current_and_goal_hand_poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the current and goal hand poses."""
        curr_hand_pos, curr_hand_ori = math_utils.subtract_frame_transforms(
            self.robot.data.root_state_w[..., :3],
            self.robot.data.root_state_w[..., 3:7],
            self.robot.data.body_state_w[:, self.target_hand_idx, :3],
            self.robot.data.body_state_w[:, self.target_hand_idx, 3:7],
        )
        curr_hand_pose = torch.cat((
            curr_hand_pos,
            math_utils.axis_angle_from_quat(curr_hand_ori)
        ), dim=-1)
        goal_hand_pose = self.command
        return curr_hand_pose, goal_hand_pose   # (N, 6), (N, 6)


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
        curr_hand_pose, goal_hand_pose = self.current_and_goal_hand_poses
        self.metrics["error_hand_pos"] += torch.norm(curr_hand_pose[:, :3] - goal_hand_pose[:, :3], dim=-1)
        self.metrics["error_hand_rot"] += torch.norm(curr_hand_pose[:, 3:6] - goal_hand_pose[:, 3:6], dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the command for the given environment IDs."""
        # get the current shoulder position
        self.torso_pos_w = self.robot.data.body_state_w[env_ids, self.torso_idx, :3]
        shoulder_offset = torch.tensor(self.cfg.shoulder_offset, device=self.env.device).unsqueeze(0)
        self.shoulder_pos_w[env_ids, :3] = self.torso_pos_w + shoulder_offset

        # FIXME(OKJ): we need to sample a random goal in dexterous workspace
        # for now, we just sample a random goal within a hemisphere around the shoulder
        # sample a random goal within a hemisphere around the shoulder
        theta = -torch.rand((self.env.num_envs,), device=self.env.device) * math.pi
        phi = torch.rand((self.env.num_envs,), device=self.env.device) * math.pi
        radius = torch.rand((self.env.num_envs,), device=self.env.device) * 0.2 + 0.2 # radius between 0.2 and 0.4
        goal_pos = radius.unsqueeze(-1) * torch.stack((
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta)
        ), dim=-1)
        self.goal_hand_pose_w[env_ids, :3] = self.shoulder_pos_w[env_ids] + goal_pos
        
        rpy = torch.rand((self.env.num_envs, 3), device=self.env.device) * math.pi - math.pi / 2
        goal_ori = math_utils.quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
        self.goal_hand_pose_w[env_ids, 3:7] = goal_ori

        # transform the goal hand pose to the base frame
        goal_hand_pos, goal_hand_quat = math_utils.subtract_frame_transforms(
            self.goal_hand_pose_w[env_ids, :3],
            self.goal_hand_pose_w[env_ids, 3:7],
            self.robot.data.root_state_w[env_ids, :3],
            self.robot.data.root_state_w[env_ids, 3:7],
        )
        self.goal_hand_pose_b[env_ids, :3] = goal_hand_pos
        self.goal_hand_pose_b[env_ids, 3:7] = goal_hand_quat

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Set debug visualization implementation."""
        if debug_vis:
            if not hasattr(self, "goal_hand_pose_visualizer"):
                self.goal_hand_pose_visualizer = VisualizationMarkers(self.cfg.goal_hand_pose_visualizer_cfg)
                self.current_hand_pose_visualizer = VisualizationMarkers(self.cfg.current_hand_pose_visualizer_cfg)
                self.shoulder_pos_visualizer = VisualizationMarkers(self.cfg.shoulder_pos_visualizer_cfg)
            # set their visibility to true
            self.goal_hand_pose_visualizer.set_visibility(True)
            self.current_hand_pose_visualizer.set_visibility(True)
            self.shoulder_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_hand_pose_visualizer"):
                self.goal_hand_pose_visualizer.set_visibility(False)
                self.current_hand_pose_visualizer.set_visibility(False)
                self.shoulder_pos_visualizer.set_visibility(False)
    
    def _debug_vis_callback(self, event):
        """Debug visualization callback."""
        if not self.robot.is_initialized:
            return
        
        # visualize the goal hand pose
        goal_hand_pos = self.goal_hand_pose_w[:, :3]
        goal_hand_ori = self.goal_hand_pose_w[:, 3:7]
        self.goal_hand_pose_visualizer.visualize(goal_hand_pos, goal_hand_ori)
        
        # visualize the current hand pose
        curr_hand_pos = self.robot.data.body_state_w[:, self.target_hand_idx, :3]
        curr_hand_ori = self.robot.data.body_state_w[:, self.target_hand_idx, 3:7]
        self.current_hand_pose_visualizer.visualize(curr_hand_pos, curr_hand_ori)
        
        # visualize the shoulder position
        self.shoulder_pos_visualizer.visualize(self.shoulder_pos_w)
