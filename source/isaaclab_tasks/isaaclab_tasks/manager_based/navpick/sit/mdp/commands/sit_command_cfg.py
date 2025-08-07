# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG
from isaaclab.utils import configclass

from .sit_command import SitCommand


@configclass
class SitCommandCfg(CommandTermCfg):
    """Configuration for the uniform root xy position command generator."""

    class_type: type = SitCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    @configclass
    class Ranges:
        """
        Ranges for the sit height commands.
        """

        height: tuple[float, float] = MISSING
        """Range for the base height command (in m)."""

    ranges: Ranges = MISSING
    
    max_velocity: float = 0.1
    """Maximum velocity for the robot to reach the target position (in m/s)."""

    slow_bound: float = 0.2
    """Slow bound for the robot to reach the target position (in m)."""

    velocity_command_option: Literal["sqrt", "linear"] = "sqrt"

    goal_height_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_height")
    """The configuration for the goal height visualization marker. Defaults to FRAME_MARKER_CFG."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    goal_height_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.1, 0.1, 0.1)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.1, 0.1, 0.1)