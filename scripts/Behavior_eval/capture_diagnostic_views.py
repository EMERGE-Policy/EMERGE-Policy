#!/usr/bin/env python3
"""Load one BEHAVIOR instance and capture the three diagnostic camera views."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from omnigibson.eval.diagnostic_cameras import DiagnosticCameraRig, make_diagnostic_camera_specs
from omnigibson.eval.evaluator import BatchedEvaluator, resolve_instance_ids
from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED, seed_everything
from omnigibson.macros import gm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-index", type=int, default=0)
    parser.add_argument("--robot-config", default="dev/behavior_r1pro_openpi.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    return parser.parse_args()


def _position(entity) -> np.ndarray:
    position, _ = entity.get_position_orientation()
    return position.detach().cpu().numpy()


def main() -> int:
    args = _parse_args()
    output_dir = Path(
        args.output_dir
        or f"/data/yxy/emerge_pi05_eval/diagnostic_camera_preview_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ).resolve()

    gm.HEADLESS = True
    gm.RENDER_VIEWER_CAMERA = False
    gm.ENABLE_TRANSITION_RULES = False

    seed = seed_everything(DEFAULT_EVAL_SEED)
    robot_config = OmegaConf.load(args.robot_config)
    instance_id = resolve_instance_ids("turning_on_radio", [args.instance_index], mode="public_test")[0]
    cfg = OmegaConf.create(
        {
            "env_wrapper": {"_target_": "omnigibson.eval.wrappers.DefaultWrapper"},
            "policy_name": "local",
            "model": {"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None},
            "headless": True,
            "partial_scene_load": True,
            "max_steps": 1,
            "write_video": False,
            "mode": "public_test",
            "seed": seed,
            "num_envs": 1,
            "task": {"name": "turning_on_radio"},
            "robot": robot_config,
        }
    )

    with BatchedEvaluator(cfg) as evaluator:
        evaluator.load_batch({0: instance_id})
        accessor = evaluator.instance_eval_states[0].env_accessor
        scope = accessor.object_scope
        radio = next(entity for name, entity in scope.items() if name.startswith("radio_receiver"))
        table = next(entity for name, entity in scope.items() if name.startswith("table"))

        robot_position = _position(accessor.robot)
        table_position = _position(table)
        radio_position = _position(radio)
        specs = make_diagnostic_camera_specs(
            robot_position=robot_position,
            table_position=table_position,
            radio_position=radio_position,
        )
        rig = DiagnosticCameraRig(specs, image_width=args.image_width, image_height=args.image_height)
        rig.load()
        paths = rig.save_snapshots(output_dir)

        metadata = {
            "task": "turning_on_radio",
            "instance_index": args.instance_index,
            "instance_id": instance_id,
            "robot_position": robot_position.tolist(),
            "table_position": table_position.tolist(),
            "radio_position": radio_position.tolist(),
            "cameras": {
                spec.name: {
                    "position": spec.position.tolist(),
                    "target": spec.target.tolist(),
                    "focal_length": spec.focal_length,
                    "image": str(paths[spec.name]),
                }
                for spec in specs
            },
        }
        (output_dir / "camera_poses.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps({"output_dir": str(output_dir), "images": {k: str(v) for k, v in paths.items()}}), flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
