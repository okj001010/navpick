# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from .bodypose_command import BodyPoseCommand


@configclass
class BodyPoseCommandCfg(CommandTermCfg):
    """Configuration for the uniform root xy position command generator."""

    class_type: type = BodyPoseCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    left_hip_pitch_joint: str = "left_hip_pitch_joint"
    right_hip_pitch_joint: str = "right_hip_pitch_joint"

    @configclass
    class Ranges:
        """
        Ranges for the body pose (base height and hip pitch) commands.
        """

        base_height: tuple[float, float] = MISSING
        hip_pitch: tuple[float, float] = MISSING

    ranges: Ranges = MISSING

    # TODO(OKJ): visualization for debugging
    goal_base_visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_base_height",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        }
    )
    current_base_visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/current_base_height",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        }
    )
    # goal_left_hip_pitch_visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
    #     prim_path="/Visuals/Command/goal_left_hip_pitch"
    # )
    # goal_right_hip_pitch_visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
    #     prim_path="/Visuals/Command/goal_right_hip_pitch"
    # )
    # current_left_hip_pitch_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
    #     prim_path="/Visuals/Command/current_left_hip_pitch",
    # )
    # current_right_hip_pitch_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
    #     prim_path="/Visuals/Command/current_right_hip_pitch"
    # )
    # goal_left_hip_pitch_visualizer_cfg.markers["arrow"].scale = (0.1, 0.1, 0.1)
    # goal_right_hip_pitch_visualizer_cfg.markers["arrow"].scale = (0.1, 0.1, 0.1)
    # current_left_hip_pitch_visualizer_cfg.markers["arrow"].scale = (0.1, 0.1, 0.1)
    # current_right_hip_pitch_visualizer_cfg.markers["arrow"].scale = (0.1, 0.1, 0.1)
    
    goal_left_hip_pitch_visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_left_hip_pitch",
        markers={
            "cylinder": sim_utils.CylinderCfg(
                radius=0.03,
                height=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        }
    )
    goal_right_hip_pitch_visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_right_hip_pitch",
        markers={
            "cylinder": sim_utils.CylinderCfg(
                radius=0.03,
                height=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        }
    )
    current_left_hip_pitch_visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/current_left_hip_pitch",
        markers={
            "cylinder": sim_utils.CylinderCfg(
                radius=0.03,
                height=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        }
    )
    current_right_hip_pitch_visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/current_right_hip_pitch",
        markers={
            "cylinder": sim_utils.CylinderCfg(
                radius=0.03,
                height=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        }
    )