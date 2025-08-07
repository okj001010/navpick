# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
import torch
from typing import TYPE_CHECKING, cast

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from .commands.sit_command import SitCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Use reward terms from eetrack sitting