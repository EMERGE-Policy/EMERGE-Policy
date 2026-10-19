"""Exercise champion control semantics in its own environment, without a model."""

import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[3]
CHAMPION = ROOT / "third_party/behavior-1k-solution"
CHAMPION_PYTHON = CHAMPION / ".venv/bin/python"


class ChampionPolicyControlTest(unittest.TestCase):
    @unittest.skipUnless(CHAMPION_PYTHON.is_file(), "champion environment is not installed")
    def test_real_wrapper_and_server_preserve_stage_across_cache_invalidation(self):
        environment = dict(os.environ)
        environment.update({"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""})
        result = subprocess.run(
            [str(CHAMPION_PYTHON), "-m", "unittest", "discover", "-s",
             str(CHAMPION / "tests"), "-p", "test_policy_control.py", "-v"],
            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Ran 10 tests", result.stderr)


if __name__ == "__main__":
    unittest.main()
