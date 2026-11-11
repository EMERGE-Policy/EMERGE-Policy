"""Persistent Cosmos3 forward-dynamics service for AC-WM candidate batches."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from external_model_server.ac_wm_contract import INPUT_SCHEMA, OUTPUT_SCHEMA, SERVICE
from external_model_server.model_service.contracts import ServiceDescriptor
from external_model_server.model_service.runtime import (
    ModelServerRuntime,
    add_runtime_arguments,
    runtime_arguments,
)


class Cosmos3Adapter:
    """Load Cosmos Framework once and expose it through the shared service runtime."""

    def __init__(self, options: dict[str, Any], *, model_id: str | None = None):
        self.options = options
        self.descriptor = ServiceDescriptor(
            SERVICE,
            model_id or str(options["checkpoint_path"]),
            INPUT_SCHEMA,
            OUTPUT_SCHEMA,
            ("infer", "action_conditioned_video"),
        )
        self.setup = None
        self.pipeline = None
        self._request_count = 0

    def load(self):
        from cosmos_framework.inference.common.init import init_output_dir, init_script

        init_script()
        # Cosmos must initialize before importing inference/model configuration.
        from cosmos_framework.inference.args import OmniSetupOverrides

        self.setup = OmniSetupOverrides.model_validate(self.options).build_setup()
        init_output_dir(Path(self.setup.output_dir), resume=True, job_name="ac-wm-cosmos3-service")
        self.pipeline = self.setup.get_inference_cls().create(self.setup)

    def close(self):
        self.pipeline = None

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        candidates = request.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("request requires a non-empty candidates list")
        output_value = str(request.get("output_dir", "")).strip()
        if not output_value:
            raise ValueError("request requires output_dir")
        output_root = Path(output_value).resolve()
        specs: list[Path] = []
        candidate_by_name: dict[str, dict[str, Any]] = {}
        for item in candidates:
            if not isinstance(item, dict):
                raise TypeError("candidate entries must be objects")
            candidate_id = str(item.get("candidate_id", "")).strip()
            input_spec = Path(str(item.get("input_spec", ""))).resolve()
            if (
                not candidate_id
                or candidate_id in candidate_by_name
                or Path(candidate_id).name != candidate_id
                or candidate_id in {".", ".."}
            ):
                raise ValueError(
                    f"candidate_id must be a unique safe path component: {candidate_id!r}"
                )
            if not input_spec.is_file():
                raise FileNotFoundError(f"Cosmos input spec does not exist: {input_spec}")
            candidate_by_name[candidate_id] = item
            specs.append(input_spec)
        override_cls = self.setup.get_sample_overrides_cls()
        overrides = override_cls.from_files(specs, overrides=self.setup.sample_overrides)
        if len(overrides) != len(candidates):
            raise RuntimeError("Cosmos parsed a different number of samples than candidates")
        sample_args = []
        candidate_dirs: dict[str, Path] = {}
        for sample in overrides:
            candidate_id = str(sample.name)
            if candidate_id not in candidate_by_name:
                raise ValueError(
                    f"input spec sample name does not match candidate id: {candidate_id!r}"
                )
            sample_dir = output_root / candidate_id
            sample.output_dir = sample_dir
            sample.download(sample_dir / "inputs")
            candidate_dirs[candidate_id] = sample_dir
            sample_args.append(sample.build_sample(model_config=self.pipeline.model_config))
        outputs = self.pipeline.generate(sample_args)
        if len(outputs) != len(candidates):
            raise RuntimeError("Cosmos returned a mismatched sample count")
        results = []
        for sample, output in zip(sample_args, outputs, strict=True):
            candidate_id = str(sample.name)
            sample_dir = candidate_dirs[candidate_id]
            videos = sorted(sample_dir.rglob("vision.mp4"))
            status = str(getattr(output, "status", "unknown"))
            ok = status in {"success", "ok"} and bool(videos)
            results.append(
                {
                    "candidate_id": candidate_id,
                    "status": "success" if ok else "failed",
                    "video_path": str(videos[0]) if videos else None,
                    "output_dir": str(sample_dir),
                    "sample_status": status,
                    "error": None
                    if ok
                    else str(getattr(output, "message", "Cosmos produced no vision.mp4")),
                }
            )
        self._request_count += 1
        logging.getLogger(__name__).info(
            "Cosmos AC-WM request completed | request=%s candidates=%d",
            self._request_count,
            len(results),
        )
        return {"results": results, "request_count": self._request_count}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8773)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--parallelism-preset", choices=("latency", "throughput"), default="latency"
    )
    parser.add_argument("--device-memory-utilization", type=float, default=0.75)
    parser.add_argument("--no-diffusion-cache", action="store_true")
    add_runtime_arguments(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    adapter = Cosmos3Adapter(
        {
            "output_dir": args.output_dir,
            "checkpoint_path": args.checkpoint,
            "config_file": args.config_file,
            "parallelism_preset": args.parallelism_preset,
            "device_memory_utilization": args.device_memory_utilization,
            "guardrails": False,
            "diffusion_cache": not args.no_diffusion_cache,
            "keep_going": True,
            "max_num_seqs": 1,
        },
        model_id=args.model_id,
    )
    ModelServerRuntime(
        adapter, host=args.host, port=args.port, **runtime_arguments(args)
    ).serve_forever()


if __name__ == "__main__":
    main()
