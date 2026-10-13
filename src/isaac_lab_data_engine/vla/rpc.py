"""Small localhost-only NumPy RPC used to isolate LeRobot from Isaac Lab."""

from __future__ import annotations

import io
import socket
import struct
from dataclasses import dataclass
from typing import Mapping

import numpy as np


_LENGTH = struct.Struct("!Q")
DEFAULT_MAX_MESSAGE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class VLARequest:
    rgb: np.ndarray
    state: np.ndarray
    task: str
    input_interface: str = "legacy"

    def as_arrays(self) -> dict[str, np.ndarray]:
        rgb = np.asarray(self.rgb, dtype=np.uint8)
        state = np.asarray(self.state, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"rgb must be [H,W,3], got {rgb.shape}")
        if state.shape not in {(25,), (29,), (30,)} or not np.isfinite(state).all():
            raise ValueError(
                f"state must be a finite [25], [29], or [30] vector, got {state.shape}"
            )
        arrays = {"rgb": rgb, "state": state, "task": np.asarray(self.task)}
        if self.input_interface not in ("legacy", "relative4d", "joint8"):
            raise ValueError("Unknown VLA input interface")
        if self.input_interface == "relative4d":
            if state.shape != (29,): raise ValueError("Relative input must be 29-D")
            arrays["input_interface"] = np.asarray(self.input_interface)
        if self.input_interface == "joint8":
            if state.shape != (25,): raise ValueError("Joint target input must be robot-only 25-D")
            arrays["input_interface"] = np.asarray(self.input_interface)
        return arrays


@dataclass(frozen=True)
class VLAResponse:
    actions: np.ndarray
    inference_ms: float
    error: str = ""

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray]) -> "VLAResponse":
        actions = np.asarray(arrays["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 8:
            raise ValueError(f"Server returned invalid action chunk {actions.shape}")
        return cls(
            actions=actions,
            inference_ms=float(np.asarray(arrays["inference_ms"]).item()),
            error=str(np.asarray(arrays.get("error", np.asarray(""))).item()),
        )


@dataclass(frozen=True)
class VLABatchRequest:
    """Ordered environment rows; ids are echoed to prevent cross-environment routing."""
    rgb: np.ndarray
    state: np.ndarray
    tasks: list[str]
    env_ids: np.ndarray
    input_interface: str = "legacy"

    def as_arrays(self):
        rgb = np.asarray(self.rgb, dtype=np.uint8)
        state = np.asarray(self.state, dtype=np.float32)
        ids = np.asarray(self.env_ids, dtype=np.int64)
        n = len(ids)
        if ids.ndim != 1 or n < 1 or len(set(ids.tolist())) != n:
            raise ValueError("Batch environment ids must be unique and nonempty")
        if rgb.ndim != 4 or rgb.shape[0] != n or rgb.shape[-1] != 3:
            raise ValueError("Batch RGB must be [B,H,W,3]")
        if state.ndim != 2 or state.shape[0] != n or len(self.tasks) != n:
            raise ValueError("Batch row counts must match")
        for i in range(n):
            VLARequest(rgb[i], state[i], self.tasks[i], self.input_interface).as_arrays()
        return dict(rgb=rgb, state=state, tasks=np.asarray(self.tasks), env_ids=ids,
                    input_interface=np.asarray(self.input_interface), batch=np.asarray(True))


@dataclass(frozen=True)
class VLABatchResponse:
    actions: np.ndarray
    inference_ms: float
    env_ids: np.ndarray


def encode_arrays(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def decode_arrays(payload: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def receive_message(connection: socket.socket, max_bytes: int = DEFAULT_MAX_MESSAGE_BYTES) -> bytes:
    header = _receive_exact(connection, _LENGTH.size)
    (size,) = _LENGTH.unpack(header)
    if size > max_bytes:
        raise ValueError(f"RPC message of {size} bytes exceeds limit {max_bytes}")
    return _receive_exact(connection, int(size))


def send_message(connection: socket.socket, payload: bytes) -> None:
    connection.sendall(_LENGTH.pack(len(payload)))
    connection.sendall(payload)


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("VLA RPC connection closed early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class VLAClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, timeout_s: float = 30.0) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)

    def predict(self, request: VLARequest) -> VLAResponse:
        with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as connection:
            connection.settimeout(self.timeout_s)
            send_message(connection, encode_arrays(request.as_arrays()))
            response = VLAResponse.from_arrays(decode_arrays(receive_message(connection)))
        if response.error:
            raise RuntimeError(response.error)
        return response

    def predict_batch(self, request: VLABatchRequest) -> VLABatchResponse:
        with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as connection:
            connection.settimeout(self.timeout_s)
            send_message(connection, encode_arrays(request.as_arrays()))
            arrays = decode_arrays(receive_message(connection))
        error = str(np.asarray(arrays.get("error", "")).item())
        if error:
            raise RuntimeError(error)
        actions = np.asarray(arrays["actions"], dtype=np.float32)
        ids = np.asarray(arrays["env_ids"], dtype=np.int64)
        if not np.array_equal(ids, request.env_ids):
            raise ValueError("Batch response environment order mismatch")
        if actions.ndim != 3 or actions.shape[0] != len(ids) or actions.shape[1] < 1 or actions.shape[2] != 8 or not np.isfinite(actions).all():
            raise ValueError("Invalid batch action tensor")
        return VLABatchResponse(actions, float(arrays["inference_ms"]), ids)
