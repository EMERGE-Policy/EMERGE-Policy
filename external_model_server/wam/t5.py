"""Cosmos-compatible online T5 encoding and generated embedding cache."""

from __future__ import annotations

import fcntl
import os
import pickle
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

T5_MAX_LENGTH = 512
T5_HIDDEN_SIZE = 1024
T5_SHAPE = (1, T5_MAX_LENGTH, T5_HIDDEN_SIZE)
CACHE_SCHEMA = "Emerge.t5_embedding_cache.v1"


def validate_embedding(value: Any) -> np.ndarray:
    embedding = np.asarray(value, dtype=np.float32)
    if embedding.shape != T5_SHAPE:
        raise ValueError(f"T5 embedding must have shape {T5_SHAPE}, got {embedding.shape}")
    if not np.isfinite(embedding).all():
        raise ValueError("T5 embedding must contain only finite values")
    return np.ascontiguousarray(embedding)


class CosmosT5Encoder:
    """Match the official Cosmos T5 tokenizer, padding, and tail masking."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        revision: str,
        cache_dir: str,
        device: str,
        local_files_only: bool,
    ) -> None:
        import torch
        from transformers import T5EncoderModel, T5TokenizerFast

        common = {
            "cache_dir": cache_dir or None,
            "local_files_only": local_files_only,
            "revision": revision,
        }
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name_or_path, **common)
        self.model = T5EncoderModel.from_pretrained(model_name_or_path, **common).to(device)
        self.model.eval()
        self.device = str(next(self.model.parameters()).device)
        self.metadata = {
            "name": model_name_or_path,
            "revision": str(getattr(self.model.config, "_commit_hash", None) or revision),
            "max_length": T5_MAX_LENGTH,
            "dtype": str(next(self.model.parameters()).dtype),
        }
        self._torch = torch

    def encode(self, text: str) -> np.ndarray:
        batch = self.tokenizer.batch_encode_plus(
            [text],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=T5_MAX_LENGTH,
            return_length=True,
            return_offsets_mapping=False,
        )
        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)
        with self._torch.inference_mode():
            embedding = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
        length = int(attention_mask.sum().item())
        array = embedding.float().cpu().numpy().copy()
        array[:, length:] = 0
        return validate_embedding(array)


class T5EmbeddingCache:
    """Store online embeddings separately with bounded locking and atomic writes."""

    def __init__(self, path: str | Path, *, lock_timeout: float = 30.0) -> None:
        self.path = Path(path).expanduser()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.lock_timeout = max(0.1, float(lock_timeout))
        self._thread_lock = threading.Lock()

    def get(self, text: str):
        with self._thread_lock, self._file_lock():
            payload = self._load_unlocked()
            value = payload["entries"].get(text)
            if value is None:
                return None, {}
            return validate_embedding(value), dict(payload["models"].get(text) or {})

    def put(self, text: str, embedding: Any, *, model: dict[str, Any]) -> None:
        normalized = validate_embedding(embedding)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_lock, self._file_lock():
            payload = self._load_unlocked()
            payload["entries"][text] = normalized
            payload["models"][text] = dict(model)
            fd, temporary = tempfile.mkstemp(
                prefix=self.path.name + ".",
                suffix=".tmp",
                dir=self.path.parent,
            )
            try:
                with os.fdopen(fd, "wb") as file:
                    pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)

    def _load_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema": CACHE_SCHEMA, "entries": {}, "models": {}}
        with self.path.open("rb") as file:
            payload = pickle.load(file)  # noqa: S301 - user-owned local cache
        if not isinstance(payload, dict) or payload.get("schema") != CACHE_SCHEMA:
            raise ValueError(f"invalid generated T5 cache: {self.path}")
        if not isinstance(payload.get("entries"), dict) or not isinstance(
            payload.get("models"), dict
        ):
            raise ValueError(f"invalid generated T5 cache contents: {self.path}")
        return payload

    def _file_lock(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        file = self.lock_path.open("a+b")
        cache = self

        class Lock:
            def __enter__(self_nonlocal):
                deadline = time.monotonic() + cache.lock_timeout
                while True:
                    try:
                        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        return file
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            file.close()
                            raise TimeoutError(
                                f"timed out locking generated T5 cache: {cache.lock_path}"
                            )
                        time.sleep(0.05)

            def __exit__(self_nonlocal, exc_type, exc, traceback):
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)
                file.close()

        return Lock()
