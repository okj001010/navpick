# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from .base_command import BaseCommand


@configclass
class BaseCommandCfg(CommandTermCfg):
    """Configuration for the base command generator."""

    class_type: type = BaseCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    
    def __post_init__(self):
        """Post initialization."""
        # set the resampling time range to infinity to avoid resampling
        self.resampling_time_range = (math.inf, math.inf)