"""Array encoding shared with OpenPI; decoding validates untrusted buffers."""

import math
from typing import Any

import msgpack
import numpy as np

from .contracts import MAX_MESSAGE_BYTES


def pack_message(value: Any) -> bytes:
    return msgpack.packb(value, default=_encode, use_bin_type=True)


def unpack_message(payload: bytes) -> Any:
    if not isinstance(payload, bytes) or len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("Expected a bounded binary model message")
    return msgpack.unpackb(payload, object_hook=_decode, raw=False)


def _dtype(value):
    dtype = np.dtype(value)
    if dtype.kind not in "biufSU" or dtype.itemsize == 0:
        raise ValueError(f"Unsupported ndarray dtype: {dtype}")
    return dtype


def _encode(value):
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {b"__ndarray__": True, b"data": array.tobytes(),
                b"dtype": _dtype(array.dtype).str, b"shape": array.shape}
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": _dtype(value.dtype).str}
    raise TypeError(f"Unsupported model value: {type(value).__name__}")


def _decode(value):
    if b"__ndarray__" in value:
        dtype, shape, data = _dtype(value[b"dtype"]), value[b"shape"], value[b"data"]
        if not isinstance(shape, (list, tuple)) or len(shape) > 32 or any(
            type(d) is not int or d < 0 for d in shape
        ) or not isinstance(data, bytes) or math.prod(shape) * dtype.itemsize != len(data):
            raise ValueError("ndarray shape, dtype and buffer length disagree")
        return np.ndarray(buffer=data, dtype=dtype, shape=shape)
    if b"__npgeneric__" in value:
        return _dtype(value[b"dtype"]).type(value[b"data"])
    return value
