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
from . import bodypose_joint_names, non_bodypose_joint_names
from ..base.g1_spawn_info import G1_CFG

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
def bodypose_joint_pos_rel(env):
    return mdp.partial_joint_pos_rel(env, joint_names=bodypose_joint_names)

def bodypose_joint_vel_rel(env):
    return mdp.partial_joint_vel_rel(env, joint_names=bodypose_joint_names)


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
            func=bodypose_joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        partial_joint_vel = ObsTerm(
            func=bodypose_joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"}, noise=Unoise(n_min=-0.01, n_max=0.01))
        body_pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "body_pose"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=bodypose_joint_names, scale=0.5, use_default_offset=True)
    fix_joint_pos = mdp.FixJointPositionActionCfg(asset_name="robot", joint_names=non_bodypose_joint_names, use_default_offset=True)


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    body_pose = mdp.BodyPoseCommandCfg(
        asset_name="robot",
        ranges=mdp.BodyPoseCommandCfg.Ranges(
            base_height=(0.3, 0.75),
            hip_pitch=(-60.0, 0.0), # front hip pitch (in degrees)
        ),
        # FIXME(OKJ): Does resampling multiple times in a single episode is better?
        resampling_time_range=(5.0, 5.0), # avoid resampling during the episode
        debug_vis= True,
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    # default reward
    termination_penalty = RewTerm(
        func=mdp.is_terminated_term,
        params={"term_names": ["pelvis_below_minimum", "bad_pelvis_ori"]},
        weight=-200.0,
    )
    
    # task reward
    body_pose = RewTerm(func=mdp.body_pose_reward, weight=1.0, params={"command_name": "body_pose"})
    
    # behavioral rewards
    waist_roll_error = RewTerm(func=mdp.waist_roll_error, weight=1.0)
    leg_pos_symmetry = RewTerm(
        func=mdp.leg_pos_symmetry,
        weight=-0.5,
        params={
            "left_leg_joint_names": "^left_(hip|knee|ankle).*_joint$",
            "right_leg_joint_names": "^right_(hip|knee|ankle).*_joint$",
        }
    )
    leg_force_symmetry = RewTerm(
        func=mdp.leg_force_symmetry,
        weight=-0.2,
        params={
            "left_leg_joint_names": "^left_(hip|knee|ankle).*_joint$",
            "right_leg_joint_names": "^right_(hip|knee|ankle).*_joint$",
        }
    )
    contact_ground = RewTerm(
        func=mdp.contact_ground,
        weight=1.0,
        params={
            # consider only ground contacts
            "max_force": 100.0,
            "left_contact_link_name": "left_ankle_roll_link",
            "right_contact_link_name": "right_ankle_roll_link",
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        }
    )

    # regularization
    action_acc_l2 = RewTerm(func=mdp.action_acc_l2, weight=-0.01)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    collision_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-5.0,
        params={
            # consider only self collisions (not ground)
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!.*_ankle_roll_link$).*"),
        }
    )
    default_joint_error = RewTerm(func=mdp.default_joint_error, weight=0.2, params={"joint_names": bodypose_joint_names})


@configclass
class EventCfg:
    """Configuration for events."""
    
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )
    
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

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    pelvis_below_minimum = DoneTerm(func=mdp.pelvis_below_minimum, params={"minimum_height": 0.25})
    bad_pelvis_ori = DoneTerm(func=mdp.bad_pelvis_ori, params={"limit_euler_angle": [1.5, 1.5]})


##
# Environment configuration
##


@configclass
class G1BodyPoseEnvCfg(ManagerBasedRLEnvCfg):
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
        self.scene.robot = G1_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )


@configclass
class G1BodyPoseEnvCfgPlay(G1BodyPoseEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False