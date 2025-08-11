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


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # FIXME (OKJ): I think we can remove these terms
        # foot_pose = ObsTerm(func=foot_pose_in_robot_root_frame)
        # hand_pose = ObsTerm(
        #     func=hand_pose_in_robot_root_frame,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot"),
        #         "right_hand_cfg": SceneEntityCfg("robot", body_names="right_rubber_hand"),
        #     },
        # )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"}, noise=Unoise(n_min=-0.01, n_max=0.01))
        sit_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "sit_command"})
        pelvis_height = ObsTerm(func=mdp.pelvis_height)
        # FIXME (OKJ): I think we can remove these terms
        # prev_pelvis_height = ObsTerm(func=mdp.prev_pelvis_height)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    sit_command = mdp.SitCommandCfg(
        asset_name="robot",
        ranges=mdp.SitCommandCfg.Ranges(height=(0.3, 0.7)),
        resampling_time_range=(5.0, 5.0), # avoid resampling during the episode
        max_velocity=0.1,
        velocity_command_option="linear",
        debug_vis= True,
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    # Default terms
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    
    termination_penalty = RewTerm(
        func=mdp.is_terminated_term,
        params={"term_keys": ["pelvis_below_minimum", "bad_pelvis_ori"]},
        weight=-50.0,
    )

    ### Regualization terms
    rel_torques_l2 = RewTerm(
        func=mdp.rel_joint_torques_l2,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    energy = RewTerm(
        func=mdp.energy,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        weight=-0.0002,
    )

    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.0e-8,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)

    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                    ".*_hip_yaw_joint",
                    ".*_hip_roll_joint",
                    "waist_.*",
                ],
            )
        },
    )

    torso_upright = RewTerm(
        func=mdp.keep_torso_upright,
        weight=-0.3,
    )

    follow_command_vel_z = RewTerm(func=mdp.follow_command_vel_z, weight=-0.2, params={"command_name": "hands_pose"})

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    standing_still_four_contact_points_v2 = RewTerm(
        func=mdp.standing_still_four_contact_points_v2,
        weight=0.1,
        params={
            "command_name": "hands_pose",
            "asset_cfg": SceneEntityCfg("robot"),
            "left_foot_sensor_cfg": SceneEntityCfg("contact_left_foot"),
            "right_foot_sensor_cfg": SceneEntityCfg("contact_right_foot"),
        },
    )

    zmp_supp_dist_v2 = RewTerm(
        func=mdp.zmp_supp_dist_v2,
        weight=0.5,
        params={
            "sigma": 10,
            "command_name": "hands_pose",
            "asset_cfg": SceneEntityCfg("robot"),
            "left_foot_sensor_cfg": SceneEntityCfg("contact_left_foot"),
            "right_foot_sensor_cfg": SceneEntityCfg("contact_right_foot"),
        },
    )

    ankle_parallel = RewTerm(
        func=mdp.ankle_parallel,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


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
    bad_pelvis_ori = DoneTerm(func=mdp.bad_pelvis_ori, params={"limit_euler_angle": [0.9, 1.0]})


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
        self.episode_length_s = 10.0
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
        
        # Contact sensors
        self.scene.contact_left_foot = mdp.ContactSensorExtraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
            filter_prim_paths_expr=["/World/ground/GroundPlane/CollisionPlane"],
            update_period=0.0,
            history_length=6,
            debug_vis=True,
            max_contact_data_count=8 * 4096,
        )
        self.scene.contact_right_foot = mdp.ContactSensorExtraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
            filter_prim_paths_expr=["/World/ground/GroundPlane/CollisionPlane"],
            update_period=0.0,
            history_length=6,
            debug_vis=True,
            max_contact_data_count=8 * 4096,
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