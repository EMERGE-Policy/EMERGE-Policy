from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from robot.plugins import register_plugin


class LightNavSonicPluginTest(unittest.TestCase):
    def test_manifest_registers_and_driver_loads(self) -> None:
        plugin_root = Path(__file__).resolve().parents[3] / "integrations/lightnav_sonic_emerge"
        with tempfile.TemporaryDirectory() as plugin_home, tempfile.TemporaryDirectory() as workspace:
            with patch.dict(os.environ, {"Emerge_PLUGIN_HOME": plugin_home}):
                spec = register_plugin(plugin_root)
                self.assertEqual(spec.driver_name, "lightnav_sonic_g1")

                from robot.drivers import load_driver

                driver = load_driver(
                    "lightnav_sonic_g1",
                    workspace=workspace,
                    elastic_band_release_file=None,
                )
                try:
                    self.assertEqual(driver.ROBOT_ID, "g1_sonic_001")
                finally:
                    driver.close()
