"""Checks for the persistent local Transformers SAM3 server path."""

from __future__ import annotations

import unittest

import numpy as np

from external_model_server.localization_types import TargetMask, TargetMaskResult
from external_model_server.sam3_server import SAM3InferenceService, build_arg_parser
from external_model_server.sam3_transformers_backend import _targets_from_result


class TransformersSAM3ResultTest(unittest.TestCase):
    def test_best_instances_are_ranked_and_keep_masks(self) -> None:
        masks = np.zeros((3, 4, 5), dtype=np.float32)
        masks[0, 0:2, 0:2] = 1.0
        masks[1, 1:4, 2:5] = 1.0
        masks[2, 0, 0] = 1.0
        targets = _targets_from_result(
            object_key="radio",
            prompt="radio receiver",
            result={
                "scores": np.array([0.7, 0.95, 0.2], dtype=np.float32),
                "masks": masks,
                "boxes": np.array(
                    [[0, 0, 2, 2], [2, 1, 5, 4], [0, 0, 1, 1]],
                    dtype=np.float32,
                ),
            },
            max_instances=2,
        )

        self.assertEqual([target.instance_index for target in targets], [1, 2])
        self.assertAlmostEqual(targets[0].confidence, 0.95, places=5)
        self.assertEqual(targets[0].area_pixels, 9)
        self.assertEqual(targets[0].bbox_2d, [2, 1, 5, 4])
        self.assertEqual(targets[1].area_pixels, 4)

    def test_missing_detection_has_the_existing_wire_contract(self) -> None:
        target = _targets_from_result(
            object_key="radio",
            prompt="radio receiver",
            result={},
            max_instances=2,
        )[0]
        self.assertFalse(target.found)
        self.assertIsNone(target.mask)
        self.assertEqual(target.instance_index, 1)


class _FakeBackend:
    def __init__(self) -> None:
        self.preloads = 0
        self.calls = 0

    def preload(self) -> None:
        self.preloads += 1

    def segment_views_batched(self, views):
        self.calls += 1
        return [
            TargetMaskResult(
                view_name=view_name,
                image_shape=rgb.shape[:2],
                targets=[
                    TargetMask(
                        object_keys[0],
                        prompts[0],
                        True,
                        0.9,
                        np.ones(rgb.shape[:2], dtype=bool),
                        [0, 0, rgb.shape[1] - 1, rgb.shape[0] - 1],
                        int(rgb.shape[0] * rgb.shape[1]),
                        1,
                    )
                ],
            )
            for rgb, view_name, object_keys, prompts, _max_instances in views
        ]


class SAM3InferenceServiceTest(unittest.TestCase):
    def test_backend_is_preloaded_once_and_response_stays_compatible(self) -> None:
        backend = _FakeBackend()
        service = SAM3InferenceService(
            {"backend": "transformers", "model_path": "/local/sam3"},
            backend=backend,
        )
        request = {
            "views": [{"name": "head", "rgb": np.zeros((3, 4, 3), dtype=np.uint8)}],
            "targets": [{"object_key": "radio", "prompt": "radio receiver"}],
            "max_instances": 2,
        }

        first = service.infer(request)
        second = service.infer(request)

        self.assertEqual(backend.preloads, 1)
        self.assertEqual(backend.calls, 2)
        self.assertEqual(first["views"][0]["targets"][0]["object_key"], "radio")
        self.assertEqual(first.keys(), second.keys())
        self.assertEqual(service.metadata["backend"], "transformers")

    def test_cli_accepts_the_transformers_backend(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "--model-path",
                "/local/sam3",
                "--backend",
                "transformers",
                "--confidence-threshold",
                "0.3",
                "--mask-threshold",
                "0.4",
            ]
        )
        self.assertEqual(args.backend, "transformers")
        self.assertEqual(args.confidence_threshold, 0.3)
        self.assertEqual(args.mask_threshold, 0.4)


if __name__ == "__main__":
    unittest.main()
