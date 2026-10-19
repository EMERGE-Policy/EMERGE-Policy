"""Test the serial Bash orchestrator with a fake runner, never real episodes."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/Behavior_eval/run_radio_agent_public10.sh"
FAKE = '''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[2:]
index = int(args[args.index('--instance-index') + 1])
with Path(os.environ['TEST_CALLS']).open('a') as stream:
    stream.write(json.dumps(args) + '\\n')
if index == int(os.environ.get('TEST_INTERRUPT_AT', '-1')):
    raise SystemExit(130)
root = Path(args[args.index('--output-dir') + 1])
(root / 'json').mkdir(parents=True)
success = index % 2 == 0
(root / 'json' / f'turning_on_radio_{301 + index}_0.json').write_text(json.dumps({
    'task': 'turning_on_radio', 'instance_id': 301 + index, 'rollout_id': 0,
    'success': success, 'steps': 10, 'q_score': {'final': float(success)}}))
(root / 'agent_episode_result.json').write_text(json.dumps({
    'task': 'turning_on_radio', 'instance_index': index, 'success': success,
    'episode_terminated': True, 'infrastructure_error': None,
    'evaluator_returncode': 0, 'exit_code': 0}))
'''


class RadioAgentPublic10Test(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fake = self.root / "fake_python"
        self.fake.write_text(FAKE)
        self.fake.chmod(0o755)
        self.calls = self.root / "calls.jsonl"
        self.output = self.root / "outputs"
        self.env = {**os.environ, "EMERGE_AGENT_PYTHON": str(self.fake),
                    "EMERGE_RADIO_OUTPUT_DIR": str(self.output), "TEST_CALLS": str(self.calls)}

    def run_batch(self, *args):
        return subprocess.run(["bash", str(SCRIPT), *args], env=self.env,
                              capture_output=True, text=True, timeout=20)

    def recorded(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []

    def test_runs_ten_independent_episodes_and_resumes_even_task_failures(self):
        first = self.run_batch()
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        calls = self.recorded()
        self.assertEqual([int(a[a.index("--instance-index") + 1]) for a in calls], list(range(10)))
        self.assertEqual(len({a[a.index("--output-dir") + 1] for a in calls}), 10)
        for args in calls:
            self.assertIn("--no-headless", args)
            self.assertIn("--write-video", args)
            self.assertNotIn("--max-steps", args)
            self.assertNotIn("--navigation-only", args)
        self.assertIn('"successes": 5', first.stdout)
        second = self.run_batch()
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertEqual(len(self.recorded()), 10)

    def test_interrupt_exit_stops_entire_batch(self):
        self.env["TEST_INTERRUPT_AT"] = "1"
        result = self.run_batch()
        self.assertEqual(result.returncode, 130, result.stdout + result.stderr)
        self.assertEqual(len(self.recorded()), 2)

    def test_partial_directory_is_preserved_not_skipped_as_complete_or_overwritten(self):
        partial = self.output / "instance_301"
        partial.mkdir(parents=True)
        marker = partial / "partial_video.mp4"
        marker.write_bytes(b"incomplete")
        result = self.run_batch("--headless")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(marker.read_bytes(), b"incomplete")
        self.assertEqual(len(self.recorded()), 9)
        self.assertTrue(all("--headless" in args for args in self.recorded()))

    def test_wrong_instance_result_is_not_treated_as_completed(self):
        self.assertEqual(self.run_batch().returncode, 0)
        path = self.output / "instance_301/json/turning_on_radio_301_0.json"
        value = json.loads(path.read_text())
        value["instance_id"] = 320
        path.write_text(json.dumps(value))
        result = self.run_batch()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(len(self.recorded()), 10)


if __name__ == "__main__":
    unittest.main()
