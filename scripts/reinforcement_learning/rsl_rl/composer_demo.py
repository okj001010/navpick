# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to demonstrate an interactive demo with the G1 composer.
(locomotion + body-pose adjustment + handreach)
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# local imports
import cli_args # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates an interactive demo with the H1 rough terrain environment."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch

import carb
import omni

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


class ComposerDemo:
    """This class provides an interactive demo for the H1 rough terrain environment.
    It loads a pre-trained checkpoint for the Isaac-Velocity-Rough-H1-v0 task, trained with RSL RL
    and defines a set of keyboard commands for directing motion of selected robots.

    A robot can be selected from the scene through a mouse click. Once selected, the following
    keyboard controls can be used to control the robot:

    * UP: go forward
    * LEFT: turn left
    * RIGHT: turn right
    * DOWN: stop
    * C: switch between third-person and perspective views
    * ESC: exit current third-person view"""

    def __init__(self):
        """Initializes environment config designed for the interactive model and sets up the environment,
        loads pre-trained checkpoints, and registers keyboard events."""
        # parse configuration
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.episode_length_s = 1000000
        
        # wrap around environment for rsl-rl
        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device

        self.action = torch.zeros(1, 14, device=self.device)
        self.set_up_keyboard()

    def set_up_keyboard(self):
        """Sets up interface for keyboard input and registers the desired keys for control."""
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        self._key_to_transition = {
            "KEY_1": 0,
            "KEY_2": 1,
            "KEY_3": 2,
        }
        self._key_to_control = {
            "W": (3, 1.0),
            "S": (3, -1.0),
            "A": (4, -1.0),
            "D": (4, 1.0),
            "Q": (5, 1.0),
            "R": (5, -1.0),
        }
        self.activated = torch.zeros(14, dtype=torch.bool, device=self.device)

    def _on_keyboard_event(self, event):
        """Checks for a keyboard event and assign the corresponding command control depending on key pressed."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._key_to_transition:
                index = self._key_to_transition[event.input.name]
                for i in range(3):
                    if i == index:
                        self.action[0, i] = 1.0
                    else:
                        self.action[0, i] = 0.0
            if event.input.name in self._key_to_control:
                index, value = self._key_to_control[event.input.name]
                if self.activated[index]:
                    self.action[0, index] = 0.0
                    self.activated[index] = False
                else:
                    self.activated[index] = True
                    self.action[0, index] = value


def main():
    """Main function."""
    demo_composer = ComposerDemo()
    demo_composer.env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            # action based on keyboard input
            action = demo_composer.action
            demo_composer.env.step(action)


if __name__ == "__main__":
    main()
    simulation_app.close()
