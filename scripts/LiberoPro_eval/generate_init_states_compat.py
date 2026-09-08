#!/usr/bin/env python3
"""Run the official LIBERO-Pro init generator with Emerge's robosuite shim."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: generate_init_states_compat.py OFFICIAL_SCRIPT [ARGS...]")
    official_script = Path(sys.argv[1]).resolve()
    source_path = official_script.parents[1]
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

    from robot.mujoco_simulation.mujoco_env import RobosuiteCompatibility

    RobosuiteCompatibility.install()
    official = runpy.run_path(str(official_script), run_name="_libero_pro_init_generator")
    RobosuiteCompatibility.patch_libero_robot_models()

    sys.argv = [str(official_script), *sys.argv[2:]]
    args = official["parse_args"]()
    official["generate_init_states"](
        bddl_base_dir=args.bddl_base_dir,
        output_dir=args.output_dir,
        num_inits=args.num_inits,
        height=args.height,
        width=args.width,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
