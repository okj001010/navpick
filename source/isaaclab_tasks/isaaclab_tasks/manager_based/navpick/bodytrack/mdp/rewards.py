# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, cast

from .commands.bodytrack_command import BodyTrackCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# FIXME(OKJ): track body command