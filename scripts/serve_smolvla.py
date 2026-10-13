"""Serve a fine-tuned SmolVLA checkpoint to Isaac Lab over localhost."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaac_lab_data_engine.vla.rpc import (  # noqa: E402
    VLARequest,
    VLABatchRequest,
    decode_arrays,
    encode_arrays,
    receive_message,
    send_message,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional fixed model/noise RNG seed for reproducible RPC diagnostics.")
    parser.add_argument("--profile-memory", action="store_true",
                        help="Log CUDA allocator and global free memory on request 1 and every 25 requests.")
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.0,
        help="Optional CUDA allocator limit; 0 disables the limit.",
    )
    return parser.parse_args()


class SmolVLARuntime:
    def __init__(self, checkpoint: Path, device: str, execution_horizon: int) -> None:
        import torch
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.smolvla import SmolVLAPolicy

        self.torch = torch
        self.device = torch.device(device)
        # Prefer a local checkpoint on Windows. It also guarantees that the
        # inference model and its normalization statistics stay versioned.
        self.policy = SmolVLAPolicy.from_pretrained(str(checkpoint.resolve()))
        self.policy.to(self.device).eval()
        features = self.policy.config.input_features
        self.state_dim = int(features["observation.state"].shape[0])
        self.action_dim = int(self.policy.config.output_features["action"].shape[0])
        self.relative_interface = self.action_dim == 4
        self.joint_interface = False
        contract_file = checkpoint / "interface_contract.json"
        if contract_file.is_file():
            saved = json.loads(contract_file.read_text(encoding="utf-8-sig"))
            if saved.get("version") == "vla_joint_execution_aligned_v23":
                from isaac_lab_data_engine.vla.joint_interface import contract
                if saved != contract() or self.state_dim != 25 or self.action_dim != 8:
                    raise ValueError("Invalid joint target checkpoint contract")
                self.joint_interface = True
        if self.relative_interface:
            from isaac_lab_data_engine.vla.relative_interface import contract
            saved = json.loads((checkpoint / "interface_contract.json").read_text(encoding="utf-8"))
            if saved != contract() or self.state_dim != 29:
                raise ValueError("Missing/mismatched relative state/action interface contract")
        self.policy.config.n_action_steps = min(
            int(execution_horizon), int(self.policy.config.chunk_size)
        )
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            str(checkpoint.resolve()),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

    def predict(self, request: VLARequest) -> np.ndarray:
        return self.predict_batch(VLABatchRequest(
            request.rgb[None], request.state[None], [request.task], np.asarray([0]), request.input_interface
        ))[0]

    def predict_batch(self, request: VLABatchRequest) -> np.ndarray:
        request.as_arrays()
        torch = self.torch
        expected_interface = "joint8" if self.joint_interface else "relative4d" if self.relative_interface else "legacy"
        if request.input_interface != expected_interface:
            raise ValueError("Input feature semantics do not match the checkpoint interface")
        rgb = np.asarray(request.rgb, dtype=np.uint8)
        image = torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).float().div_(255.0)
        state = torch.from_numpy(np.asarray(request.state, dtype=np.float32))
        if state.shape[-1] != self.state_dim:
            raise ValueError("Observation width does not match the checkpoint contract")
        batch = {
            "observation.images.table": image,
            "observation.state": state,
            "task": request.tasks,
        }
        processed = self.preprocess(batch)
        with torch.inference_mode():
            action_chunk = self.policy.predict_action_chunk(processed)
            action_chunk = self.postprocess(action_chunk)
        actions = action_chunk[:, : self.policy.config.n_action_steps].detach().float().cpu().numpy()
        if actions.ndim != 3 or actions.shape[0] != len(request.env_ids) or actions.shape[2] != self.action_dim:
            raise ValueError(
                f"Checkpoint action contract mismatch; received {actions.shape}"
            )
        if self.relative_interface:
            from isaac_lab_data_engine.vla.relative_interface import expand_relative_action
            actions = np.stack([expand_relative_action(row) for row in actions])
        return actions.astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("The unauthenticated VLA RPC server may only bind to localhost")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    if args.execution_horizon < 1:
        raise ValueError("execution-horizon must be positive")
    if args.seed is not None:
        if not 0 <= args.seed <= np.iinfo(np.uint32).max:
            raise ValueError("seed must fit in an unsigned 32-bit integer")
        import torch
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.gpu_memory_fraction:
        import torch

        if not 0.0 < args.gpu_memory_fraction <= 1.0:
            raise ValueError("gpu-memory-fraction must be in (0,1]")
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)

    runtime = SmolVLARuntime(args.checkpoint, args.device, args.execution_horizon)
    print(
        "SMOLVLA_SERVER_READY="
        + json.dumps(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "host": args.host,
                "port": args.port,
                "device": args.device,
                "execution_horizon": args.execution_horizon,
                "seed": args.seed,
                "gpu_memory_fraction": args.gpu_memory_fraction,
                "state_dim": runtime.state_dim,
                "model_action_dim": runtime.action_dim,
                "input_interface": "joint8" if runtime.joint_interface else "relative4d" if runtime.relative_interface else "legacy",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        request_count = 0
        while True:
            connection, address = server.accept()
            with connection:
                start = time.perf_counter()
                arrays = {}
                try:
                    arrays = decode_arrays(receive_message(connection))
                    interface = str(np.asarray(arrays.get("input_interface", "legacy")).item())
                    if bool(np.asarray(arrays.get("batch", False)).item()):
                        request = VLABatchRequest(arrays["rgb"], arrays["state"],
                            arrays["tasks"].tolist(), arrays["env_ids"], interface)
                        actions = runtime.predict_batch(request)
                    else:
                        request = VLARequest(arrays["rgb"], arrays["state"],
                            str(np.asarray(arrays["task"]).item()), interface)
                        actions = runtime.predict(request)
                    error = ""
                    request_count += 1
                    if args.profile_memory and runtime.device.type == "cuda" and (request_count == 1 or request_count % 25 == 0):
                        free_bytes, total_bytes = runtime.torch.cuda.mem_get_info(runtime.device)
                        print("SMOLVLA_CUDA_MEMORY=" + json.dumps({
                            "request": request_count,
                            "batch_size": int(arrays["state"].shape[0]) if arrays["state"].ndim == 2 else 1,
                            "allocated_mib": runtime.torch.cuda.memory_allocated(runtime.device) / 2**20,
                            "reserved_mib": runtime.torch.cuda.memory_reserved(runtime.device) / 2**20,
                            "peak_allocated_mib": runtime.torch.cuda.max_memory_allocated(runtime.device) / 2**20,
                            "peak_reserved_mib": runtime.torch.cuda.max_memory_reserved(runtime.device) / 2**20,
                            "global_free_mib": free_bytes / 2**20,
                            "global_total_mib": total_bytes / 2**20,
                        }), flush=True)
                except BaseException as exception:
                    traceback.print_exc()
                    actions = np.empty((0, 8), dtype=np.float32)
                    error = f"{type(exception).__name__}: {exception}"
                send_message(
                    connection,
                    encode_arrays(
                        {
                            "actions": actions,
                            "inference_ms": np.asarray(
                                (time.perf_counter() - start) * 1000.0, dtype=np.float32
                            ),
                            "error": np.asarray(error),
                            "env_ids": np.asarray(arrays.get("env_ids", []), dtype=np.int64),
                        }
                    ),
                )


if __name__ == "__main__":
    main()
