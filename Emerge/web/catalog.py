"""Read the driver's scene catalog in a separate interpreter, without loading a scene."""

import json
import sys
from pathlib import Path

from robot.controller import load_driver_config
from robot.drivers import load_driver

if __name__ == "__main__":
    driver = load_driver(sys.argv[1], gui=False, **load_driver_config(Path(sys.argv[2])))
    try:
        catalog = driver.get_scene_catalog()
        Path(sys.argv[3]).write_text(json.dumps(catalog), encoding="utf-8")
    finally:
        driver.close()
