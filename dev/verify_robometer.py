"""Mock-only interface/regression checks. Run: python -m unittest dev.verify_robometer -v.

No model weights, GPU inference or robot task is simulated as real verification.
The multipart requests are decoded, so assertions cover the actual wire payload.
"""
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from email import policy
from email.parser import BytesParser
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import numpy as np

from external_model_server.cosmos_policy_server import (
    CosmosPolicyInferenceService, CosmosPolicyRuntime, build_arg_parser,
)
from external_model_server.wam.robometer import DEFAULT_MODEL_PATH, RobometerClient
from robot.wam.planner import CosmosWAMPlanner
from robot.wam.protocol import ProtocolError, validate_candidate_response


def parse_multipart(request):
    message = BytesParser(policy=policy.default).parsebytes(
        ("Content-Type: " + request.headers["content-type"] + "\r\n\r\n").encode()
        + request.content
    )
    return {part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
            for part in message.iter_parts()}


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.current = np.zeros((8, 8, 3), dtype=np.uint8)
        self.current[0, :, 0] = 250  # asymmetric RGB, detects accidental flipping/BGR
        self.futures = [np.full_like(self.current, i + 1) for i in range(3)]
        self.calls = []

    def client(self, traces=None, status=200, info_override=None):
        def handle(request):
            self.calls.append(request)
            if request.url.path == "/model_info":
                info = {"model_path": DEFAULT_MODEL_PATH, "experiment_config": {"model": {
                    "train_progress_head": True, "use_multi_image": True,
                    "use_per_frame_progress_token": True,
                }, "data": {"shuffle_progress_frames": False, "use_multi_image": True,
                            "use_per_frame_progress_token": True}}}
                return httpx.Response(200, json=info_override or info)
            return httpx.Response(status, content=json.dumps({"outputs_progress": {
                "progress_pred": traces if traces is not None else [[0.9, 0.1], [0.0, 0.8], [0.6, 0.3]],
            }}))
        client = RobometerClient("http://robometer.test")
        client._http.close()
        client._http = httpx.Client(transport=httpx.MockTransport(handle))
        self.addCleanup(client.close)
        return client

    def test_multipart_distinct_future_and_terminal_score(self):
        scores, info = self.client().score(self.current, self.futures, "full task", [0, 1, 2])
        self.assertEqual(scores, [0.1, 0.8, 0.3])  # not sum, success, or first-frame score
        self.assertEqual(info["score_semantics"], "final_frame_task_progress")
        posts = [r for r in self.calls if r.method == "POST"]
        self.assertEqual(len(posts), 1)
        fields = parse_multipart(posts[0])
        self.assertEqual(fields["use_frame_steps"], b"false")
        for i in range(3):
            sample = json.loads(fields[f"sample_{i}"])
            traj = sample["trajectory"]
            self.assertEqual(traj["task"], "full task")
            self.assertEqual(traj["id"], f"candidate_{i}")
            self.assertEqual(traj["metadata"]["subsequence_length"], 2)
            key = traj["frames"]["__numpy_file__"]
            array = np.load(io.BytesIO(fields[key]), allow_pickle=False)
            self.assertEqual(array.dtype, np.uint8)
            self.assertEqual(array.shape, (2, 8, 8, 3))
            np.testing.assert_array_equal(array[0], self.current)
            np.testing.assert_array_equal(array[1], self.futures[i])

    def runtime(self, client=None):
        args = build_arg_parser().parse_args([
            "--policy-checkpoint", "unused", "--base-model-dir", "unused",
            "--dataset-stats", "unused", "--t5-embeddings", "unused",
        ])
        runtime = CosmosPolicyRuntime(args)
        runtime.robometer = client
        runtime.model, runtime.dataset_stats = object(), {}
        runtime.cfg = SimpleNamespace(flip_images=True, num_queries_best_of_n=1, randomize_seed=False,
                                     use_variance_scale=False)
        runtime._resolve_embedding = lambda text: (np.zeros((1, 2, 3)), {})
        generated = []
        def get_action(cfg, model, stats, observation, embedding, **kwargs):
            index = kwargs["seed"] - 10
            generated.append(kwargs["generate_future_state_and_value_in_parallel"])
            # Stand in for the upstream prepared images, with a distinctive pixel
            # edit to verify this result is used rather than raw request images.
            primary = observation["primary_image"].copy()
            primary[2, 3] = [51, 52, 53]
            return {"actions": np.full((16, 7), index, np.float32),
                    "value_prediction": [0.99, 0.01, 0.2][index],
                    "future_image_predictions": {"future_image": self.futures[index]},
                    "all_camera_images": [observation["wrist_image"], primary]}
        runtime.cosmos_utils = SimpleNamespace(get_action=get_action, t5_text_embeddings_cache={"full task": 1})
        request = {"task_instruction": "full task", "conditioning_text": "phase text",
                   "conditioning_mode": "phase", "primary_image": self.current,
                   "wrist_image": self.current, "proprio": np.zeros(9), "seed": 10,
                   "candidate_request": {"num_candidates": 3, "score_mode": "robometer"}}
        return runtime, request, generated

    def test_runtime_and_planner_use_robometer_instead_of_cosmos(self):
        runtime, request, generated = self.runtime(self.client())
        response = runtime.infer(request)
        self.assertEqual(generated, [True, True, True])
        self.assertEqual([c["score"] for c in response["candidates"]], [0.1, 0.8, 0.3])
        validate_candidate_response(response)
        mock_client = SimpleNamespace(infer=lambda **kwargs: response)
        plan = CosmosWAMPlanner({"num_candidates": 3, "score_mode": "robometer"}).plan(
            mock_client, {}, "full task", seed=10,
        )
        self.assertEqual(plan.search_decision["selected_index"], 1)
        self.assertEqual(plan.search_decision["scoring"]["backend"], "robometer")
        np.testing.assert_array_equal(plan.actions, np.ones((16, 7), np.float32))
        fields = parse_multipart([r for r in self.calls if r.method == "POST"][0])
        self.assertEqual(json.loads(fields["sample_0"])["trajectory"]["task"], "full task")
        frames = np.load(io.BytesIO(fields["sample_0_trajectory_frames"]), allow_pickle=False)
        expected = np.flipud(self.current).copy(); expected[2, 3] = [51, 52, 53]
        np.testing.assert_array_equal(frames[0], expected)
        self.assertNotIn("_future_frame", response["candidates"][0])

    def test_joint_value_unchanged(self):
        runtime, request, generated = self.runtime()
        request["candidate_request"]["score_mode"] = "joint_value"
        response = runtime.infer(request)
        self.assertEqual([c["score"] for c in response["candidates"]], [0.99, 0.01, 0.2])
        self.assertNotIn("scoring", response)
        plan = CosmosWAMPlanner({"num_candidates": 3, "score_mode": "joint_value"}).plan(
            SimpleNamespace(infer=lambda **kw: response), {}, "full task", seed=10)
        self.assertEqual(plan.search_decision["selected_index"], 0)

    def test_wrist_uses_current_and_future_from_same_camera(self):
        client = self.client()
        client.view = "wrist"
        runtime, request, _ = self.runtime(client)
        original = runtime.cosmos_utils.get_action
        request["wrist_image"] = np.full_like(self.current, 17)
        def with_wrist(*args, **kwargs):
            result = original(*args, **kwargs)
            result["future_image_predictions"]["future_wrist_image"] = np.full_like(self.current, 29)
            return result
        runtime.cosmos_utils.get_action = with_wrist
        response = runtime.infer(request)
        self.assertEqual(response["scoring"]["view"], "wrist")
        fields = parse_multipart([r for r in self.calls if r.method == "POST"][0])
        pair = np.load(io.BytesIO(fields["sample_0_trajectory_frames"]), allow_pickle=False)
        np.testing.assert_array_equal(pair[0], np.full_like(self.current, 17))
        np.testing.assert_array_equal(pair[1], np.full_like(self.current, 29))

    def test_invalid_images_and_candidate_order(self):
        client = self.client()
        for image in (self.current.astype(np.float32), self.current[:, :, :1],
                      self.current[None, ...]):
            with self.subTest(shape=image.shape, dtype=image.dtype), self.assertRaises(ProtocolError):
                client.score(image, self.futures, "task", [0, 1, 2])
        with self.assertRaisesRegex(ProtocolError, "contiguous"):
            client.score(self.current, self.futures, "task", [1, 0, 2])
        with self.assertRaisesRegex(ProtocolError, "image shape"):
            client.score(self.current, [np.zeros((9, 9, 3), np.uint8)], "task", [0])

    def test_missing_futures_and_configuration_are_errors(self):
        runtime, request, _ = self.runtime()
        with self.assertRaisesRegex(ProtocolError, "requires --robometer-url"):
            runtime.infer(request)
        runtime.robometer = self.client()
        runtime.cosmos_utils.get_action = lambda *a, **kw: {"actions": np.zeros((16, 7))}
        with self.assertRaisesRegex(ProtocolError, "missing future_image"):
            runtime.infer(request)

    def test_invalid_outputs_fail_without_fallback(self):
        for traces in ([], [[0.1, 0.2], [0.1, 0.2]], [[0.1]], [[0.1, float("nan")]],
                       [[0.1, float("inf")]], [[0.1, True]], [[0.1, "0.5"]]):
            with self.subTest(traces=traces):
                c = self.client(traces=traces)
                with self.assertRaises(ProtocolError):
                    c.score(self.current, self.futures[:1], "task", [0])

    def test_http_failure_and_timeout(self):
        with self.assertRaisesRegex(ProtocolError, "HTTP failure"):
            self.client(status=503).score(self.current, self.futures, "task", [0, 1, 2])
        client = self.client()
        client._http.close()
        def timed_out(request):
            raise httpx.ReadTimeout("mock timeout", request=request)
        client._http = httpx.Client(transport=httpx.MockTransport(timed_out))
        with self.assertRaisesRegex(ProtocolError, "timed out"):
            client.score(self.current, self.futures, "task", [0, 1, 2])
        def disconnected(request):
            raise httpx.ConnectError("mock connection failure", request=request)
        client._http.close()
        client._http = httpx.Client(transport=httpx.MockTransport(disconnected))
        with self.assertRaisesRegex(ProtocolError, "HTTP failure"):
            client.score(self.current, self.futures, "task", [0, 1, 2])

    def test_model_identity_and_unsupported_preprocessing(self):
        for info in ({"model_path": "wrong"}, {"model_path": DEFAULT_MODEL_PATH,
                     "experiment_config": {"model": {"train_progress_head": True, "use_multi_image": False}}},
                     {"model_path": DEFAULT_MODEL_PATH, "experiment_config": {"model": {
                         "train_progress_head": True, "use_multi_image": True,
                         "use_per_frame_progress_token": True}, "data": {"use_multi_image": False}}}):
            with self.assertRaises(ProtocolError):
                self.client(info_override=info).score(self.current, self.futures, "task", [0, 1, 2])

    def test_cross_environment_batch_is_rejected(self):
        runtime, request, _ = self.runtime(self.client())
        with self.assertRaisesRegex(ProtocolError, "single-environment"):
            runtime.infer_batch([request, request])
        service = CosmosPolicyInferenceService(runtime)
        wire = {"schema": "emerge.wam.request.v2", "task_instruction": "full task", "seed": 10,
                "observation": {"primary_image": self.current, "wrist_image": self.current,
                                "proprio": np.zeros(9)}, "candidate_request": request["candidate_request"]}
        from robot.wam.protocol import REQUEST_SCHEMA
        wire["schema"] = REQUEST_SCHEMA
        self.assertNotEqual(service.batch_key(wire), service.batch_key(dict(wire)))


class EvaluationDependencyTests(unittest.TestCase):
    """Mock dependency failures before LLM/simulation, not model performance."""

    def setUp(self):
        from scripts.Libero_eval import eval_libero_agent as evaluation
        self.evaluation = evaluation
        self.args = evaluation._build_parser().parse_args([
            "--robometer-url", "http://robometer.test", "--robometer-model-path", DEFAULT_MODEL_PATH,
        ])
        self.config = {"wam": {"search": {"score_mode": "robometer"}}}
        self.discovery = patch.object(evaluation, "ServiceDiscovery").start()
        self.addCleanup(patch.stopall)
        self.discovery.return_value.resolve.return_value = "ws://127.0.0.1:8003"
        self.discovery.return_value.probe.side_effect = lambda endpoint, expectation: {
            "service": expectation.service, "model_id": "test", "status": "ready",
            "input_schema": expectation.input_schema, "output_schema": expectation.output_schema,
        }

    def test_preflight_rejects_absent_robometer_without_starting_agent(self):
        with patch.object(RobometerClient, "check_ready", side_effect=ProtocolError("Connection refused")):
            with self.assertRaisesRegex(RuntimeError, "no episodes started.*"):
                self.evaluation._check_model_services(self.args, self.config)

    def test_preflight_requires_perception_and_explicit_url(self):
        self.args.robometer_url = None
        self.discovery.return_value.resolve.side_effect = lambda expectation: (
            "ws://127.0.0.1:8003" if expectation.service == "cosmos_policy_wam"
            else (_ for _ in ()).throw(RuntimeError("no endpoint"))
        )
        with self.assertRaises(RuntimeError) as caught:
            self.evaluation._check_model_services(self.args, self.config)
        for text in ("vggt", "sam3", "--robometer-url"):
            self.assertIn(text, str(caught.exception))

    def test_joint_value_preflight_does_not_contact_robometer(self):
        with patch.object(RobometerClient, "check_ready") as readiness:
            checks = self.evaluation._check_model_services(self.args, {})
            readiness.assert_not_called()
        self.assertEqual(set(checks), {"cosmos_policy_wam", "vggt", "sam3"})

    def test_model_rpc_error_is_infrastructure_failure(self):
        with TemporaryDirectory() as tmp:
            action = Path(tmp) / "ACTION.md"
            action.write_text('```json\n' + json.dumps({"actions": [{
                "id": "one", "action_type": "wam_execute", "parameters": {"step": 16},
                "status": "failed", "result": "WAM execution error after 0 steps (ServiceError: Connection refused)",
            }]}) + '\n```\n')
            self.assertIn("Connection refused", self.evaluation._action_service_error(action))
            action.write_text(action.read_text().replace("ServiceError:", "TaskIncomplete:"))
            self.assertIsNone(self.evaluation._action_service_error(action))

    def test_action_document_replacement_is_not_infrastructure_failure(self):
        with patch.object(Path, "read_text", side_effect=FileNotFoundError("replacement in progress")):
            self.assertIsNone(self.evaluation._action_service_error(Path("ACTION.md")))


if __name__ == "__main__":
    unittest.main()
