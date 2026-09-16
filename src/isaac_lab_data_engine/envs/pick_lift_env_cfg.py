"""Manager-based deterministic Franka pick-and-lift environment configuration."""

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG
from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.ik_abs_env_cfg import (
    FrankaCubeLiftEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import ObjectTableSceneCfg, ObservationsCfg

from .phase1_scene import Phase1SceneCfg, TABLE_TOP_Z
from . import observations as task_observations
from . import ppo_rewards

_PHASE1_SCENE_CFG = Phase1SceneCfg(num_envs=1, env_spacing=2.5)


def configure_parallel_physx_capacity(env_cfg: "PickLiftEnvCfg", num_envs: int) -> int:
    """Size the GPU aggregate-pair buffer for a vectorized Franka scene.

    The upstream Lift task uses 16K entries. PhysX reports about three
    aggregate pairs per environment for this scene, so four per environment
    provides headroom. A power-of-two allocation avoids oversized buffers at
    ordinary scales while preserving contacts above 4096 environments.
    """

    required = max(16 * 1024, 4 * num_envs)
    capacity = 1 << (required - 1).bit_length()
    env_cfg.sim.physx.gpu_total_aggregate_pairs_capacity = capacity
    return capacity


@configclass
class PickLiftSceneCfg(ObjectTableSceneCfg):
    """Phase 1 scene assets plus the manager-based lift task entities."""

    table_camera: CameraCfg = _PHASE1_SCENE_CFG.table_camera.copy()


@configclass
class PickLiftObservationsCfg(ObservationsCfg):
    """State observations required by the project task definition."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        end_effector_pose = ObsTerm(func=task_observations.end_effector_pose_in_robot_root_frame)
        object_pose = ObsTerm(func=task_observations.object_pose_in_robot_root_frame)
        object_position_in_end_effector = ObsTerm(
            func=task_observations.object_position_in_end_effector_frame
        )

        def __post_init__(self) -> None:
            super().__post_init__()
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class PickLiftIKPPOObservationsCfg(PickLiftObservationsCfg):
    """IK PPO observations with adaptive geometry/grasp phase feedback."""

    @configclass
    class PolicyCfg(PickLiftObservationsCfg.PolicyCfg):
        # The vertical-lift task has no fixed Cartesian goal command. Keeping
        # the old (0.55, 0, 0.30) command in the policy input would encourage
        # unwanted lateral transport after grasping.
        target_object_position = None
        grasp_state = ObsTerm(func=task_observations.grasp_state_features)
        adaptive_phase = ObsTerm(func=task_observations.adaptive_manipulation_phase)

    policy: PolicyCfg = PolicyCfg()


@configclass
class PickLiftEnvCfg(FrankaCubeLiftEnvCfg):
    """Deterministic absolute-IK environment used by the scripted baseline."""

    scene: PickLiftSceneCfg = PickLiftSceneCfg(num_envs=1, env_spacing=2.5)
    observations: PickLiftObservationsCfg = PickLiftObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.seed = 42
        self.scene.lazy_sensor_update = False
        self.num_rerenders_on_reset = 2
        self.viewer.eye = (1.60, -2.00, 1.65)
        self.viewer.lookat = (0.25, 0.0, 0.85)

        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, TABLE_TOP_Z)
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        phase1_scene = Phase1SceneCfg(num_envs=self.scene.num_envs, env_spacing=self.scene.env_spacing)
        self.scene.table = phase1_scene.table.copy()
        self.scene.plane = phase1_scene.ground.copy()
        self.scene.light = phase1_scene.light.copy()
        self.scene.object = phase1_scene.object.copy()
        self.scene.ee_frame = phase1_scene.end_effector_frame.copy()
        self.scene.table_camera = phase1_scene.table_camera.copy()

        # Keep Phase 2 deterministic. Domain randomization starts in Phase 4.
        self.events.reset_object_position = None
        self.observations.policy.enable_corruption = False

        # Fixed lift target expressed in the Franka root frame.
        self.commands.object_pose.resampling_time_range = (20.0, 20.0)
        self.commands.object_pose.ranges.pos_x = (0.55, 0.55)
        self.commands.object_pose.ranges.pos_y = (0.0, 0.0)
        self.commands.object_pose.ranges.pos_z = (0.30, 0.30)
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.ranges.yaw = (0.0, 0.0)
        self.commands.object_pose.debug_vis = False

        lift_height = TABLE_TOP_Z + 0.10
        self.rewards.lifting_object.params["minimal_height"] = lift_height
        self.rewards.object_goal_tracking.params["minimal_height"] = lift_height
        self.rewards.object_goal_tracking_fine_grained.params["minimal_height"] = lift_height
        self.terminations.object_dropping.params["minimum_height"] = TABLE_TOP_Z - 0.05

        self.episode_length_s = 12.0
        self.decimation = 2
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.device = "cuda:0"
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.render.antialiasing_mode = "DLAA"


@configclass
class PickLiftPPOEnvCfg(PickLiftEnvCfg):
    """PPO variant using the joint-position action space expected by the official agent config.

    The scripted controller deliberately uses absolute differential IK.  Isaac
    Lab only registers ``LiftCubePPORunnerCfg`` for the Franka joint-position
    task, so PPO gets a separate environment class while sharing this
    project's scene, observations, rewards, termination rules, and geometry.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, TABLE_TOP_Z)
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.actions.arm_action = lift_mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.5,
            use_default_offset=True,
        )
        # The project's formal success threshold is 10 cm above the table,
        # which is intentionally much stricter than Isaac Lab's stock lift
        # reward.  A 2 cm training gate plus a proximity-gated close reward
        # provides a curriculum: reach -> close -> lift -> track the 30 cm goal.
        # The cube center starts 3 cm above the tabletop (half-height plus a
        # 5 mm clearance), so 5 cm above the tabletop means a true 2 cm lift.
        training_lift_height = TABLE_TOP_Z + 0.05
        self.rewards.grasp_shaping = RewTerm(
            func=ppo_rewards.proximity_gated_gripper_close,
            weight=2.0,
            params={
                "distance_std": 0.06,
                "open_finger_position": 0.04,
                "robot_cfg": SceneEntityCfg("robot", joint_names=["panda_finger.*"]),
            },
        )
        self.rewards.height_progress = RewTerm(
            func=ppo_rewards.object_height_progress,
            weight=10.0,
            params={
                "start_height": TABLE_TOP_Z + 0.03,
                "target_height": TABLE_TOP_Z + 0.10,
            },
        )
        self.rewards.lifting_object.params["minimal_height"] = training_lift_height
        self.rewards.object_goal_tracking.params["minimal_height"] = training_lift_height
        self.rewards.object_goal_tracking_fine_grained.params["minimal_height"] = training_lift_height

        # The stock curriculum reaches its full -0.1 smoothness penalties at
        # 10k global steps (~417 PPO iterations).  This harder 10 cm task needs
        # longer exploration, so postpone the ramp beyond the 1500-iteration
        # baseline while retaining the small initial regularizers.
        self.curriculum.action_rate.params["num_steps"] = 50_000
        self.curriculum.joint_vel.params["num_steps"] = 50_000


@configclass
class PickLiftIKPPOEnvCfg(PickLiftEnvCfg):
    """Absolute-IK PPO task with adaptive grasp feedback and vertical lifting."""

    observations: PickLiftIKPPOObservationsCfg = PickLiftIKPPOObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Train across the complete Hard benchmark planar workspace. The
        # offsets are relative to the nominal (0.55, 0.0) object pose.
        self.events.reset_object_position = EventTerm(
            func=lift_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.19, 0.23),
                    "y": (-0.30, 0.30),
                    "z": (0.0, 0.0),
                    "yaw": (-3.141592653589793, 3.141592653589793),
                },
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
            },
        )
        # Dense progress and grasp rewards preserve the demonstrated behavior
        # while the formal sparse rewards still use the true 10 cm threshold.
        self.rewards.grasp_shaping = RewTerm(
            func=ppo_rewards.proximity_gated_gripper_close,
            weight=2.0,
            params={
                "distance_std": 0.06,
                "open_finger_position": 0.04,
                "robot_cfg": SceneEntityCfg("robot", joint_names=["panda_finger.*"]),
            },
        )
        self.rewards.height_progress = RewTerm(
            func=ppo_rewards.object_height_progress,
            weight=10.0,
            params={
                "start_height": TABLE_TOP_Z + 0.03,
                "target_height": TABLE_TOP_Z + 0.10,
            },
        )
        # The formal task only requires lifting above the table. Disable the
        # upstream reward that pulls every object laterally toward (0.55, 0),
        # which is especially destructive near the Hard workspace boundary.
        self.rewards.object_goal_tracking = None
        self.rewards.object_goal_tracking_fine_grained = None
