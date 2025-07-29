# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

from .bodypose_command import BodyPoseCommand


@configclass
class BodyPoseCommandCfg(CommandTermCfg):
    """Configuration for the uniform root xy position command generator."""

    class_type: type = BodyPoseCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    
    target_hand_name: str = MISSING
    torso_body_name: str = "torso_link"

    shoulder_offset: list[float] = [0.0039563, -0.10021, 0.24778]

    goal_hand_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/goal_hand_pose"
    )
    current_hand_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/current_hand_pose"
    )
    goal_hand_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_hand_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    shoulder_pos_visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/shoulder_pos",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        }
    )