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

from .commands.bodypose_command import BodyPoseCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# FIXME(OKJ): track body command
def body_pose_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "body_pose",
) -> torch.Tensor:
    """Reward for body-pose adjustment."""
    command = cast(BodyPoseCommand, env.command_manager.get_term(command_name))
    curr_hand_pose, goal_hand_pose = command.current_and_goal_hand_poses
    return torch.exp(-4.0 * torch.square(curr_hand_pose - goal_hand_pose).sum(dim=-1))