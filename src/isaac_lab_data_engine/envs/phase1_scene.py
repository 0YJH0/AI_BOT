"""Phase 1 scene configuration: Franka, table, object, and RGB-D camera."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


TABLE_CENTER = (0.45, 0.0, 0.40)
TABLE_SIZE = (1.40, 0.90, 0.10)
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_SIZE[2] / 2.0
OBJECT_SIZE = (0.05, 0.05, 0.05)


@configclass
class Phase1SceneCfg(InteractiveSceneCfg):
    """A deterministic single-object manipulation scene.

    Phase 1 deliberately contains no task logic, controller, reward, or
    randomization. It establishes the physical scene and verifies that the
    state and sensor interfaces needed by later phases are available.
    """

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_CENTER),
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.24, 0.18)),
            semantic_tags=[("class", "table")],
        ),
    )

    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.0, 0.0, TABLE_TOP_Z)
    robot.spawn.semantic_tags = [("class", "robot")]

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.55, 0.0, TABLE_TOP_Z + OBJECT_SIZE[2] / 2.0 + 0.005),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=OBJECT_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6,
                dynamic_friction=0.5,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.12, 0.08)),
            semantic_tags=[("class", "target_object")],
        ),
    )

    end_effector_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),
            )
        ],
        debug_vis=False,
    )

    table_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/TableCamera",
        update_period=0.0,
        height=240,
        width=320,
        data_types=["rgb", "distance_to_image_plane"],
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.80, 0.80, 0.80), intensity=2500.0),
    )

