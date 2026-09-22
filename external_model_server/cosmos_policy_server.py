"""Persistent websocket server for Cosmos Policy WAM inference.

Run this module in the Cosmos Policy environment. The Emerge runtime imports
only the lightweight WAM client and protocol; Cosmos model loading stays here.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import secrets
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from external_model_server.model_service.contracts import ServiceDescriptor
from external_model_server.model_service.runtime import (
    ModelServerRuntime,
    add_runtime_arguments,
    runtime_arguments,
)
from external_model_server.wam.batching import generate_candidates_batch
from external_model_server.wam.t5 import (
    CosmosT5Encoder,
    T5EmbeddingCache,
    validate_embedding,
)
from robot.wam.protocol import (
    REQUEST_SCHEMA,
    RESPONSE_SCHEMA,
    ProtocolError,
    ensure_finite_latency,
    make_candidate_response,
    make_response,
    validate_request,
)

logger = logging.getLogger("cosmos_policy_wam")


def _patch_local_checkpoint_resolvers(base_model_dir: Path) -> None:
    """Map Cosmos internal Hugging Face URIs to verified local files."""
    from cosmos_policy._src.imaginaire.utils import checkpoint_db
    from cosmos_policy.utils import checkpoint_utils

    local_files = {
        "hf://nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt": str(
            base_model_dir / "model-480p-16fps.pt"
        ),
        "hf://nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth": str(
            base_model_dir / "tokenizer" / "tokenizer.pth"
        ),
    }
    for path in local_files.values():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing local Cosmos base-model file: {path}")

    original_resolve = checkpoint_utils.resolve_checkpoint_path

    def get_checkpoint_by_hf(checkpoint_hf: str) -> str:
        if checkpoint_hf in local_files:
            return local_files[checkpoint_hf]
        logger.warning("Unmapped Hugging Face checkpoint requested: %s", checkpoint_hf)
        return checkpoint_hf

    def resolve_checkpoint_path(
        checkpoint_path: str, cache_dir: str | None = None
    ) -> str:
        if checkpoint_path in local_files:
            return local_files[checkpoint_path]
        return original_resolve(checkpoint_path, cache_dir=cache_dir)

    checkpoint_db.get_checkpoint_by_hf = get_checkpoint_by_hf
    checkpoint_utils.resolve_checkpoint_path = resolve_checkpoint_path
    checkpoint_db.get_checkpoint_path.cache_clear()


class CosmosPolicyRuntime:
    """Load the Cosmos model once and serve action-only inference requests."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model: Any | None = None
        self.cfg: Any | None = None
        self.cosmos_config: Any | None = None
        self.dataset_stats: dict[str, Any] | None = None
        self.cosmos_utils: Any | None = None
        self.device = "unknown"
        self.generated_t5_cache: T5EmbeddingCache | None = None
        self.t5_encoder: CosmosT5Encoder | None = None

    def load(self) -> None:
        for name in (
            "policy_checkpoint",
            "base_model_dir",
            "dataset_stats",
            "t5_embeddings",
        ):
            path = Path(getattr(self.args, name)).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"missing required path: {path}")
        if self.args.hf_home:
            os.environ.setdefault("HF_HOME", self.args.hf_home)

        _patch_local_checkpoint_resolvers(Path(self.args.base_model_dir).expanduser())
        import cosmos_policy.experiments.robot.cosmos_utils as cosmos_utils
        from cosmos_policy.experiments.robot.cosmos_utils import (
            get_model,
            init_t5_text_embeddings_cache,
            load_dataset_stats,
        )

        self.cosmos_utils = cosmos_utils
        self.cfg = SimpleNamespace(
            suite="libero",
            model_family="cosmos",
            config=self.args.config_name,
            ckpt_path=str(Path(self.args.policy_checkpoint).expanduser()),
            config_file=self.args.config_file,
            planning_model_config_name="",
            planning_model_ckpt_path="",
            use_third_person_image=True,
            num_third_person_images=1,
            use_wrist_image=True,
            num_wrist_images=1,
            use_proprio=True,
            flip_images=self.args.flip_images,
            use_variance_scale=self.args.use_variance_scale,
            use_jpeg_compression=self.args.use_jpeg_compression,
            ar_future_prediction=False,
            ar_value_prediction=False,
            ar_qvalue_prediction=False,
            num_denoising_steps_action=self.args.num_denoising_steps_action,
            num_denoising_steps_future_state=self.args.num_denoising_steps_future_state,
            num_denoising_steps_value=self.args.num_denoising_steps_value,
            num_queries_best_of_n=1,
            num_value_predictions_in_ensemble=(
                self.args.num_value_predictions_in_ensemble
            ),
            value_ensemble_aggregation_scheme=(
                self.args.value_ensemble_aggregation_scheme
            ),
            use_ensemble_value_predictions=self.args.use_ensemble_value_predictions,
            num_future_state_predictions_in_ensemble=(
                self.args.num_future_state_predictions_in_ensemble
            ),
            future_state_ensemble_aggregation_scheme=(
                self.args.future_state_ensemble_aggregation_scheme
            ),
            use_ensemble_future_state_predictions=(
                self.args.use_ensemble_future_state_predictions
            ),
            mask_current_state_action_for_value_prediction=(
                self.args.mask_current_state_action_for_value_prediction
            ),
            mask_future_state_for_qvalue_prediction=(
                self.args.mask_future_state_for_qvalue_prediction
            ),
            search_depth=self.args.search_depth,
            search_depth_value_aggregation_scheme=(
                self.args.search_depth_value_aggregation_scheme
            ),
            use_parallel_inference=False,
            available_gpus="",
            parallel_timeout=self.args.parallel_timeout,
            num_denoising_steps=self.args.num_denoising_steps_action,
            dataset_stats_path=str(Path(self.args.dataset_stats).expanduser()),
            t5_text_embeddings_path=str(Path(self.args.t5_embeddings).expanduser()),
            trained_with_image_aug=self.args.trained_with_image_aug,
            chunk_size=self.args.chunk_size,
            num_open_loop_steps=self.args.num_open_loop_steps,
            deterministic=self.args.deterministic,
            deterministic_reset=False,
            deterministic_reset_seed=None,
            seed=self.args.seed,
            randomize_seed=self.args.randomize_seed,
            unnormalize_actions=self.args.unnormalize_actions,
            normalize_proprio=self.args.normalize_proprio,
            use_ensemble_qvalue_predictions=(
                self.args.use_ensemble_qvalue_predictions
            ),
        )
        if self.cfg.search_depth != 1:
            raise ValueError(
                "Cosmos WAM currently supports search_depth=1; "
                "multi-depth planning is not exposed by this adapter"
            )
        if not 1 <= self.cfg.num_open_loop_steps <= self.cfg.chunk_size:
            raise ValueError("num_open_loop_steps must be in [1, chunk_size]")
        self.dataset_stats = load_dataset_stats(self.cfg.dataset_stats_path)
        init_t5_text_embeddings_cache(self.cfg.t5_text_embeddings_path)
        if self.args.generated_t5_cache:
            self.generated_t5_cache = T5EmbeddingCache(
                self.args.generated_t5_cache
            )
        self.model, self.cosmos_config = get_model(self.cfg)
        train_dataset = getattr(self.cosmos_config, "dataloader_train", None)
        train_dataset = getattr(train_dataset, "dataset", None)
        train_chunk_size = getattr(train_dataset, "chunk_size", None)
        if train_chunk_size is None:
            raise RuntimeError(
                "Cosmos model config does not expose the training dataset chunk_size"
            )
        if int(train_chunk_size) != self.cfg.chunk_size:
            raise ValueError(
                "Cosmos action chunk mismatch: "
                f"checkpoint={int(train_chunk_size)}, server={self.cfg.chunk_size}"
            )
        if self.args.embedding_mode == "cache-then-online":
            if self.generated_t5_cache is None:
                raise ValueError(
                    "--generated-t5-cache is required for cache-then-online mode"
                )
            self.t5_encoder = CosmosT5Encoder(
                self.args.t5_model_name_or_path,
                revision=self.args.t5_revision,
                cache_dir=self.args.t5_cache_dir,
                device=self.args.t5_device,
                local_files_only=self.args.t5_local_files_only,
            )
        self.device = str(next(self.model.parameters()).device)
        logger.info("Cosmos Policy WAM ready: device=%s", self.device)

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        observation, embedding, conditioning = self._prepare_request(request)
        candidate_request = request["candidate_request"]
        self.cfg.num_queries_best_of_n = candidate_request["num_candidates"]
        started = time.perf_counter()
        candidates = [
            self._generate_candidate(
                observation, embedding, seed=request["seed"] + index,
                index=index, score_mode=candidate_request["score_mode"],
            )
            for index in range(candidate_request["num_candidates"])
        ]
        return self._response(candidates, candidate_request, conditioning, started, 1)

    def infer_batch(self, requests: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        if not requests:
            return []
        if len(requests) == 1:
            return [self.infer(requests[0])]
        candidate_request = requests[0]["candidate_request"]
        if any(request["candidate_request"] != candidate_request for request in requests):
            raise ValueError("Batched requests must use the same candidate settings")
        prepared = [self._prepare_request(request) for request in requests]
        observations, embeddings, conditions = zip(*prepared)
        self.cfg.num_queries_best_of_n = candidate_request["num_candidates"]
        started = time.perf_counter()
        candidates = [[] for _ in requests]
        for index in range(candidate_request["num_candidates"]):
            seeds = [
                secrets.randbits(32) % 256 if self.cfg.randomize_seed else request["seed"] + index
                for request in requests
            ]
            generated = generate_candidates_batch(
                self.cfg, self.model, self.dataset_stats, self.cosmos_utils,
                observations, embeddings, seeds=seeds, index=index,
                score_mode=candidate_request["score_mode"],
                num_steps=self.args.num_denoising_steps_action,
            )
            if len(generated) != len(requests):
                raise RuntimeError("Cosmos returned an unexpected batch size")
            for items, candidate, request in zip(candidates, generated, requests):
                # Preserve the existing wire seed convention in randomize mode.
                candidate["seed"] = request["seed"] + index
                items.append(candidate)
        responses = [
            self._response(items, candidate_request, condition, started, len(requests))
            for items, condition in zip(candidates, conditions)
        ]
        logger.info("WAM batch: requests=%d candidates=%d infer_ms=%.1f",
                    len(requests), candidate_request["num_candidates"],
                    responses[0]["latency_ms"])
        return responses

    def _prepare_request(self, request):
        if self.model is None or self.cfg is None or self.dataset_stats is None:
            raise RuntimeError("model is not ready")
        instruction = request["conditioning_text"]
        embedding, conditioning = self._resolve_embedding(instruction)
        conditioning["mode"] = request["conditioning_mode"]
        conditioning["text_sha256"] = hashlib.sha256(
            instruction.encode("utf-8")
        ).hexdigest()
        observation = {
            "primary_image": np.ascontiguousarray(
                np.flipud(request["primary_image"])
                if self.cfg.flip_images
                else request["primary_image"]
            ),
            "wrist_image": np.ascontiguousarray(
                np.flipud(request["wrist_image"])
                if self.cfg.flip_images
                else request["wrist_image"]
            ),
            "proprio": request["proprio"],
        }
        return observation, embedding, conditioning

    def _response(self, candidates, candidate_request, conditioning, started, batch_size):
        latency_ms = ensure_finite_latency((time.perf_counter() - started) * 1000)
        if len(candidates) == 1 and candidate_request["score_mode"] == "none":
            response = make_response(
                candidates[0]["actions"],
                model=self.args.model_name,
                latency_ms=latency_ms,
            )
        else:
            response = make_candidate_response(
                candidates,
                score_mode=candidate_request["score_mode"],
                model=self.args.model_name,
                latency_ms=latency_ms,
            )
        response["conditioning"] = conditioning
        response["server_timing"] = {"batch_size": batch_size, "infer_ms": latency_ms}
        return response

    def _resolve_embedding(self, instruction: str):
        official = self.cosmos_utils.t5_text_embeddings_cache
        if instruction in official:
            value = official[instruction]
            if hasattr(value, "detach"):
                value = value.detach().float().cpu().numpy()
            return validate_embedding(value), {
                "source": "official_cache",
                "model": {"name": "google-t5/t5-11b", "revision": "precomputed"},
            }
        if self.generated_t5_cache is not None:
            generated, model = self.generated_t5_cache.get(instruction)
            if generated is not None:
                return generated, {"source": "generated_cache", "model": model}
        if self.args.embedding_mode != "cache-then-online" or self.t5_encoder is None:
            raise ProtocolError(
                "instruction is not present in the local T5 embedding cache: "
                f"{instruction!r}",
                code="cache_miss",
            )
        embedding = self.t5_encoder.encode(instruction)
        model = self.t5_encoder.metadata
        if self.generated_t5_cache is None:
            raise RuntimeError("generated T5 cache is not configured")
        self.generated_t5_cache.put(instruction, embedding, model=model)
        return embedding, {"source": "online_t5", "model": model}

    def _generate_candidate(
        self,
        observation: dict[str, Any],
        instruction: np.ndarray,
        *,
        seed: int,
        index: int,
        score_mode: str,
    ) -> dict[str, Any]:
        generate_value = score_mode == "joint_value"
        result = self.cosmos_utils.get_action(
            self.cfg,
            self.model,
            self.dataset_stats,
            observation,
            instruction,
            seed=seed,
            randomize_seed=self.cfg.randomize_seed,
            num_denoising_steps_action=self.args.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=generate_value,
        )
        score = None
        if score_mode == "joint_value":
            score = float(result["value_prediction"])
        elif score_mode == "q_value":
            if not self.args.enable_q_value:
                raise ProtocolError(
                    "q_value scoring is disabled", code="unsupported_score_mode"
                )
            q_result = self.cosmos_utils.get_qvalue_prediction(
                self.cfg,
                self.model,
                data_batch=result["data_batch"],
                action_sample=result["generated_latent"],
                seed=seed,
                randomize_seed=self.cfg.randomize_seed,
                num_denoising_steps_value=self.cfg.num_denoising_steps_value,
                use_ensemble_value_predictions=(
                    self.cfg.use_ensemble_value_predictions
                ),
                num_value_predictions_in_ensemble=(
                    self.cfg.num_value_predictions_in_ensemble
                ),
            )
            score = float(q_result["value_prediction"])
        return {
            "index": index,
            "seed": seed,
            "actions": result["actions"],
            "score": score,
        }


class CosmosPolicyInferenceService:
    """Adapt the Cosmos runtime to the shared websocket inference service."""

    def __init__(self, runtime: CosmosPolicyRuntime) -> None:
        self.runtime = runtime

    @property
    def descriptor(self) -> ServiceDescriptor:
        return ServiceDescriptor("cosmos_policy_wam", self.runtime.args.model_id or self.runtime.args.model_name,
                                 REQUEST_SCHEMA, RESPONSE_SCHEMA, ("infer", "batch"))

    def load(self) -> None:
        self.runtime.load()

    def close(self) -> None:
        self.runtime.model = None
        self.runtime.t5_encoder = None
        self.runtime.generated_t5_cache = None

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.runtime.infer(validate_request(request))

    def batch_key(self, request: dict[str, Any]):
        normalized = validate_request(request)
        candidate = normalized["candidate_request"]
        if not 0 <= normalized["seed"] <= 2**32 - candidate["num_candidates"]:
            raise ProtocolError("seed and candidate seeds must fit unsigned 32-bit integers")
        primary, wrist = normalized["primary_image"], normalized["wrist_image"]
        if primary.shape != wrist.shape or primary.shape[0] != primary.shape[1]:
            raise ProtocolError("Cosmos requires equally sized square primary and wrist images")
        if candidate["score_mode"] == "q_value" and not self.runtime.args.enable_q_value:
            raise ProtocolError("q_value scoring is disabled", code="unsupported_score_mode")
        # Advanced samplers keep their original singleton behavior. Uncached
        # text is isolated too, so a cache miss cannot fail other workers.
        if (
            candidate["score_mode"] == "q_value"
            or self.runtime.cfg.use_variance_scale
            or self.runtime.model.config.use_flowunipc_scheduler
            or normalized["conditioning_text"] not in self.runtime.cosmos_utils.t5_text_embeddings_cache
        ):
            return ("single", id(request))
        return ("libero", primary.shape, candidate["num_candidates"], candidate["score_mode"])

    def infer_batch(self, requests: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.runtime.infer_batch([validate_request(request) for request in requests])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cosmos Policy WAM websocket inference server"
    )
    add_runtime_arguments(parser)
    parser.add_argument("--policy-checkpoint", "--ckpt-path", required=True)
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--dataset-stats", "--dataset-stats-path", required=True)
    parser.add_argument(
        "--t5-embeddings", "--t5-text-embeddings-path", required=True
    )
    parser.add_argument(
        "--embedding-mode",
        choices=("cache-only", "cache-then-online"),
        default="cache-only",
    )
    parser.add_argument("--generated-t5-cache", default="")
    parser.add_argument("--t5-model-name-or-path", default="google-t5/t5-11b")
    parser.add_argument("--t5-revision", default="main")
    parser.add_argument("--t5-cache-dir", default="")
    parser.add_argument("--t5-device", default="cuda")
    parser.add_argument("--t5-local-files-only", action="store_true")
    parser.add_argument(
        "--config-name",
        "--config",
        dest="config_name",
        default="cosmos_predict2_2b_480p_libero__inference_only",
    )
    parser.add_argument("--config-file", default="cosmos_policy/config/config.py")
    parser.add_argument(
        "--num-denoising-steps-action",
        "--denoising-steps",
        dest="num_denoising_steps_action",
        type=int,
        default=5,
    )
    parser.add_argument("--num-denoising-steps-future-state", type=int, default=1)
    parser.add_argument("--num-denoising-steps-value", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--num-open-loop-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--parallel-timeout", type=float, default=15.0)
    parser.add_argument("--search-depth", type=int, default=1)
    parser.add_argument(
        "--search-depth-value-aggregation-scheme", default="use_last_value"
    )
    parser.add_argument(
        "--flip-images", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--use-jpeg-compression", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--trained-with-image-aug", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--normalize-proprio", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--unnormalize-actions", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--randomize-seed", action="store_true")
    parser.add_argument("--use-variance-scale", action="store_true")
    parser.add_argument("--use-ensemble-value-predictions", action="store_true")
    parser.add_argument("--use-ensemble-future-state-predictions", action="store_true")
    parser.add_argument("--num-value-predictions-in-ensemble", type=int, default=1)
    parser.add_argument(
        "--num-future-state-predictions-in-ensemble", type=int, default=1
    )
    parser.add_argument("--value-ensemble-aggregation-scheme", default="average")
    parser.add_argument(
        "--future-state-ensemble-aggregation-scheme", default="average"
    )
    parser.add_argument(
        "--mask-current-state-action-for-value-prediction", action="store_true"
    )
    parser.add_argument(
        "--mask-future-state-for-qvalue-prediction", action="store_true"
    )
    parser.add_argument("--use-ensemble-qvalue-predictions", action="store_true")
    parser.add_argument("--enable-q-value", action="store_true")
    parser.add_argument("--model-name", default="Cosmos-Policy-LIBERO-Predict2-2B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--batch-wait-ms", type=float, default=10.0)
    parser.add_argument("--hf-home", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    args = build_arg_parser().parse_args(argv)
    if args.max_batch_size < 1:
        raise SystemExit("--max-batch-size must be at least 1")
    if not math.isfinite(args.batch_wait_ms) or args.batch_wait_ms < 0:
        raise SystemExit("--batch-wait-ms must be finite and non-negative")
    for name in (
        "num_denoising_steps_action",
        "num_denoising_steps_future_state",
        "num_denoising_steps_value",
        "chunk_size",
        "search_depth",
        "num_value_predictions_in_ensemble",
        "num_future_state_predictions_in_ensemble",
    ):
        if getattr(args, name) <= 0:
            option = "--" + name.replace("_", "-")
            raise SystemExit(f"{option} must be positive")
    if not 1 <= args.num_open_loop_steps <= args.chunk_size:
        raise SystemExit("--num-open-loop-steps must be in [1, --chunk-size]")
    if args.parallel_timeout <= 0:
        raise SystemExit("--parallel-timeout must be positive")
    runtime = CosmosPolicyRuntime(args)
    ModelServerRuntime(
        CosmosPolicyInferenceService(runtime), host=args.host, port=args.port,
        max_batch_size=args.max_batch_size, batch_wait_ms=args.batch_wait_ms,
        **runtime_arguments(args),
    ).serve_forever()


if __name__ == "__main__":
    main()
