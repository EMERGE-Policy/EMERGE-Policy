"""Asset/selection/output checks; episode execution and services are mocked.

Run: python -m unittest dev.verify_libero_plus -v
These checks do not run LIBERO simulation, an LLM or real model inference.
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import stat
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

from scripts.LiberoPlus_eval.install_assets import asset_members, install
from scripts.LiberoPlus_eval import eval_libero_plus_agent as plus


class PlusTests(unittest.TestCase):
    def test_install_nested_official_layout_and_resume(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, destination = root / "assets.zip", root / "installed"
            with zipfile.ZipFile(source, "w") as z:
                z.writestr("inspire/hdd/project/release/assets/scenes/table.xml", "<mujoco/>")
                z.writestr("inspire/hdd/project/release/assets/textures/", "")
                z.writestr("README.txt", "outside the assets subtree")
            with redirect_stdout(io.StringIO()):
                install(source, destination)
                stamp = (destination / "scenes/table.xml").stat().st_mtime_ns
                install(source, destination)
            self.assertEqual((destination / "scenes/table.xml").read_text(), "<mujoco/>")
            self.assertEqual((destination / "scenes/table.xml").stat().st_mtime_ns, stamp)
            self.assertTrue((destination / "textures").is_dir())
            self.assertFalse((destination / "README.txt").exists())

    def test_reject_unsafe_ambiguous_and_symlink_assets(self):
        for names in (("assets/../escape",), ("one/assets/a", "two/assets/b"), ("assets/link",)):
            with self.subTest(names=names), TemporaryDirectory() as tmp:
                path = Path(tmp) / "bad.zip"
                with zipfile.ZipFile(path, "w") as z:
                    for name in names:
                        info = zipfile.ZipInfo(name)
                        if name.endswith("link"):
                            info.external_attr = (stat.S_IFLNK | 0o777) << 16
                        z.writestr(info, "../outside")
                with zipfile.ZipFile(path) as z, self.assertRaises(ValueError):
                    asset_members(z)

    def selection(self):
        categories = plus._load_task_categories(plus.REPO_ROOT / "third_party/libero_plus")
        return plus._select_task_ids_by_dimension(categories, list(plus.DIMENSION_ORDER),
                                                   episodes_per_dimension=50, seed=7)

    def test_fifty_distinct_tasks_each_and_robometer_plus_source(self):
        selection = self.selection()
        self.assertEqual(sum(len(ids) for suites in selection.values() for ids in suites.values()), 350)
        for suites in selection.values():
            keys = [(suite, task) for suite, ids in suites.items() for task in ids]
            self.assertEqual(len(keys), 50)
            self.assertEqual(len(set(keys)), 50)
        config = json.loads((plus.REPO_ROOT / "dev/libero_plus_robometer_eval.json").read_text())
        self.assertEqual(config["libero"]["libero_source_path"], "third_party/libero_plus")
        self.assertEqual(config["wam"]["search"], {"num_candidates": 4, "score_mode": "robometer"})

    def test_plus_main_uses_shared_dependency_check_before_mock_execution(self):
        selection = self.selection()
        specs = [{"key": f"{plus._dimension_slug(dim)}:{suite}:task-{task}:trial-0", "dimension": dim}
                 for dim, suites in selection.items() for suite, ids in suites.items() for task in ids]
        with TemporaryDirectory() as tmp:
            argv = ["eval_libero_plus_agent.py", "--dimension", "all", "--count", "50",
                    "--driver-config", "dev/libero_plus_robometer_eval.json", "--output-dir", tmp]
            with (patch.object(sys, "argv", argv),
                  patch.object(plus, "_ensure_imagemagick_env"),
                  patch.object(plus, "_install_robosuite_compat"),
                  patch.object(plus.base, "_prepare_libero_config"),
                  patch.object(plus, "_build_dimension_specs", return_value=specs),
                  patch.object(plus, "_validate_bddl_sources"),
                  patch.object(plus.base, "_check_model_services", return_value={}) as checks,
                  patch.object(plus.base, "_iter_episode_results", return_value=iter(())) as execution,
                  redirect_stdout(io.StringIO())):
                self.assertEqual(plus.main(), 0)
            checks.assert_called_once()
            self.assertEqual(checks.call_args.args[1]["wam"]["search"]["score_mode"], "robometer")
            self.assertEqual(len(execution.call_args.args[0]), 350)
            self.assertTrue((Path(tmp) / "libero_plus_grid.md").exists())
            self.assertEqual(json.loads((Path(tmp) / "evaluation_plan.json").read_text())["total_episodes"], 350)

    def test_dimension_outputs_keep_task_success_and_exclude_infrastructure_errors(self):
        results = {
            "a": {"dimension": "Robot Initial States", "suite": "libero_object", "success": True,
                  "termination_reason": "success"},
            "b": {"dimension": "Light Conditions", "suite": "libero_goal", "success": False,
                  "termination_reason": "agent_exit_0:stop"},
            "c": {"dimension": "Light Conditions", "suite": "libero_goal", "success": False,
                  "termination_reason": "infrastructure_error:unavailable"},
        }
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            plus._write_dimension_summary(root, results, {})
            plus._write_dimension_grid(root, results, {})
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["episodes"], 2)
            self.assertEqual(summary["successes"], 1)
            self.assertEqual(summary["dimensions"]["Light Conditions"]["episodes"], 1)
            self.assertTrue((root / "robot_initial_states/summary.json").exists())


if __name__ == "__main__":
    unittest.main()
