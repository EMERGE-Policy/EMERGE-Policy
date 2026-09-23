"""Run the standalone furnished-room SONIC Sim2Sim used by this bridge.

This module has no HumanClawAgent dependency.  It writes only a derived MJCF
under ``outputs/`` and injects its path into SONIC's runtime config in memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import threading
import time

from .procthor_scene import compose_procthor_g1_scene
from .room_scene import compose_lightnav_room_scene


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sonic-root", type=Path, required=True)
    parser.add_argument(
        "--environment",
        choices=("simple", "procthor"),
        default="simple",
        help="simple = bridge-owned test room; procthor = bundled mujoco_demo val_2 room",
    )
    parser.add_argument("--base-scene", type=Path)
    parser.add_argument(
        "--procthor-scene",
        type=Path,
        default=None,
        help=(
            "ProcTHOR MJCF used with --environment procthor; when omitted, "
            "resolve it below --lightnav-root"
        ),
    )
    parser.add_argument(
        "--lightnav-root",
        type=Path,
        help="official LightNav-0 checkout containing mujoco_demo assets",
    )
    parser.add_argument(
        "--output-scene",
        type=Path,
        default=None,
    )
    parser.add_argument("--run", action="store_true", help="start the blocking MuJoCo loop")
    parser.add_argument("--gui", action="store_true", help="show SONIC's third-person MuJoCo GUI")
    parser.add_argument("--publish-ego", action="store_true", help="publish head_camera as ego_view")
    parser.add_argument(
        "--elastic-band-release-file",
        type=Path,
        help=(
            "keep SONIC's safety tether until this file is created; use the same "
            "path with runner --elastic-band-release-file"
        ),
    )
    parser.add_argument("--camera-port", type=int, default=5555)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sonic_root = args.sonic_root.resolve()
    if not sonic_root.is_dir():
        raise FileNotFoundError(f"SONIC root does not exist: {sonic_root}")
    base_scene = args.base_scene or (
        sonic_root / "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
    )
    if args.output_scene is None:
        filename = (
            "scene_43dof_lightnav_room.xml"
            if args.environment == "simple"
            else "scene_43dof_lightnav_procthor.xml"
        )
        output_scene = (Path("outputs/lightnav_sonic_room") / filename).resolve()
    else:
        output_scene = args.output_scene.resolve()
    if args.environment == "simple":
        compose_lightnav_room_scene(base_scene, output_scene)
        expected_names = {
            "head_camera": "camera",
            "lightnav_overview": "camera",
            "lightnav_table_top": "geom",
        }
    else:
        procthor_scene = args.procthor_scene
        if procthor_scene is None:
            if args.lightnav_root is None:
                raise ValueError(
                    "--environment procthor requires --lightnav-root or --procthor-scene"
                )
            procthor_scene = (
                args.lightnav_root
                / "mujoco_demo/vln_mujoco/assets/scenes/procthor-10k-val/val_2_ceiling.xml"
            )
        compose_procthor_g1_scene(base_scene, procthor_scene, output_scene)
        expected_names = {
            "head_camera": "camera",
            "lightnav_procthor_anchor": "body",
        }

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(output_scene))
    object_types = {
        "camera": mujoco.mjtObj.mjOBJ_CAMERA,
        "geom": mujoco.mjtObj.mjOBJ_GEOM,
        "body": mujoco.mjtObj.mjOBJ_BODY,
    }
    missing = [
        name
        for name, type_name in expected_names.items()
        if mujoco.mj_name2id(model, object_types[type_name], name) < 0
    ]
    if missing:
        raise RuntimeError(f"generated room is missing expected MJCF names: {missing}")
    print(f"validated {args.environment} room: {output_scene} (ngeom={model.ngeom})")
    print(f"initial G1 root pose: qpos[:7]={list(model.qpos0[:7])}")
    if not args.run:
        print("dry run complete; pass --run to start SONIC Sim2Sim")
        return 0

    # SONIC's BaseSimulator itself initializes its Unitree channel. Calling
    # init_channel here as well can trigger duplicate CycloneDDS setup.
    sys.path.insert(0, str(sonic_root))
    from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory

    config = SimLoopConfig(
        interface="sim",
        enable_onscreen=args.gui,
        enable_offscreen=args.publish_ego,
        enable_image_publish=args.publish_ego,
    )
    wbc_config = config.load_wbc_yaml()
    wbc_config["ROBOT_SCENE"] = str(output_scene)
    wbc_config["ENV_NAME"] = "lightnav_sonic_room"
    wbc_config["EGO_CAMERA_NAME"] = "head_camera"
    robot_model = instantiate_g1_robot_model()
    del robot_model
    simulator = SimulatorFactory.create_simulator(
        config=wbc_config,
        env_name="default",
        onscreen=args.gui,
        offscreen=args.publish_ego,
        enable_image_publish=args.publish_ego,
    )
    # The ElasticBand keeps the G1 upright while SONIC's TensorRT deployment
    # warms up. The bridge releases it only after the runner observed yaw
    # feedback, so the policy is already in control of the robot.
    elastic_band = getattr(getattr(simulator, "sim_env", None), "elastic_band", None)
    if elastic_band is not None and args.elastic_band_release_file is not None:
        release_file = args.elastic_band_release_file.resolve()
        if release_file.exists():
            raise FileExistsError(
                f"elastic-band release file already exists: {release_file}; "
                "remove this stale signal before starting the simulator"
            )

        def release_elastic_band() -> None:
            while getattr(simulator, "_running", False):
                if release_file.exists():
                    elastic_band.enable = False
                    print("released SONIC ElasticBand: free-root navigation is enabled")
                    return
                time.sleep(0.05)

        threading.Thread(target=release_elastic_band, daemon=True).start()
        print(f"SONIC ElasticBand is armed; waiting for release signal: {release_file}")
    SimulatorFactory.start_simulator(
        simulator,
        as_thread=False,
        enable_image_publish=args.publish_ego,
        camera_port=args.camera_port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
