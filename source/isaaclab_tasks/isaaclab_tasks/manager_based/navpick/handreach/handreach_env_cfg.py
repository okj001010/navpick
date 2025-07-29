# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from . import handreach_joint_names, non_handreach_joint_names
from ..base.g1_spawn_info import G1_FIXED_CFG

##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a humanoid robot."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )

    # robot
    robot: ArticulationCfg = MISSING
    
    # contact sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


# Wrapper for the partial joint position & velocity function
def handreach_joint_pos_rel(env):
    return mdp.partial_joint_pos_rel(env, joint_names=handreach_joint_names)

def handreach_joint_vel_rel(env):
    return mdp.partial_joint_vel_rel(env, joint_names=handreach_joint_names)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        partial_joint_pos = ObsTerm(
            func=handreach_joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        partial_joint_vel = ObsTerm(
            func=handreach_joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"}, noise=Unoise(n_min=-0.01, n_max=0.01))
        hand_reach_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "hand_reach"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=handreach_joint_names, scale=0.5, use_default_offset=True)
    fix_joint_pos = mdp.FixJointPositionActionCfg(asset_name="robot", joint_names=non_handreach_joint_names, use_default_offset=True)


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    hand_reach = mdp.HandReachCommandCfg(
        asset_name="robot",
        target_hand_name="right_rubber_hand",
        # FIXME(OKJ): Does resampling multiple times in a single episode is better?
        resampling_time_range=(5.0, 5.0), # avoid resampling during the episode
        debug_vis= True,
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    # task reward
    hand_reach = RewTerm(func=mdp.hand_reach_reward, weight=1.0, params={"command_name": "hand_reach"})
    
    # regularization
    action_acc_l2 = RewTerm(func=mdp.action_acc_l2, weight=-0.01)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    collision_penalty = RewTerm(func=mdp.collision_penalty, weight=-5.0)
    default_joint_error = RewTerm(func=mdp.default_joint_error, weight=0.2, params={"joint_names": handreach_joint_names})


@configclass
class EventCfg:
    """Configuration for events."""
    
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # NOTE: Since the root is fixed, we don't need to check for other termination.
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class G1HandReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the goal conditioned locomotion environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=5.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        
        # Scene
        self.scene.robot = G1_FIXED_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )


@configclass
class G1HandReachEnvCfgPlay(G1HandReachEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False