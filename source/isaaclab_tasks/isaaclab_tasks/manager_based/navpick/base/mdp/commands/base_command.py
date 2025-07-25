# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators that does nothing."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .base_command_cfg import BaseCommandCfg


class BaseCommand(CommandTerm):
    """Command generator for base commands."""

    cfg: BaseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: BaseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Base command.

        Raises:
            RuntimeError: No command is generated. Always raises this error.
        """
        raise RuntimeError("BaseCommandTerm does not generate any commands.")
    
    """
    Operations
    """
        
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        return {}

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Set debug visualization implementation."""
        pass
    
    def _debug_vis_callback(self, event):
        """Debug visualization callback."""
        pass
