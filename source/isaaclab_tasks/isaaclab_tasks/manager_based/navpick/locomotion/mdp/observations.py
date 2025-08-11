# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING, Optional, List

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def leg_phase(
    env: ManagerBasedRLEnv,
    period: float,
) -> torch.Tensor:
    """Returns the leg phase command for the robot.

    Args:
        env: Manager-based environment
        asset_cfg: Configuration for the scene entity

    Returns:
        Tensor of shape (num_envs, 2) containing the left and right leg phases.
    """
    # Compute the normalized phase (0 <= phase < 1)
    if hasattr(env, 'episode_length_buf'):
        episode_length_buf = env.episode_length_buf
    else:
        episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = (episode_length_buf * env.step_dt) % period / period
    return torch.cat(
            (torch.sin(2 * math.pi * phase).unsqueeze(-1),
             torch.cos(2 * math.pi * phase).unsqueeze(-1)),
            dim=-1
        )