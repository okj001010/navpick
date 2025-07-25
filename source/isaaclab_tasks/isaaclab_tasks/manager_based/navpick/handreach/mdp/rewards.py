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

from .commands.handreach_command import HandReachCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def hand_reach_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "hand_reach",
) -> torch.Tensor:
    """Reward for reaching the hand to the target."""
    command = cast(HandReachCommand, env.command_manager.get_term(command_name))
    curr_hand_pose, goal_hand_pose = command.current_and_goal_hand_poses
    return torch.exp(-4.0 * torch.square(curr_hand_pose - goal_hand_pose).sum(dim=-1))