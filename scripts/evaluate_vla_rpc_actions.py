"""Evaluate a running VLA RPC server against aligned LeRobot actions.

This is an imitation sanity check on the exported dataset, not a simulator
success-rate benchmark. Use held-out Isaac seeds for the final benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaac_lab_data_engine.vla.rpc import VLAClient, VLARequest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "exports/lerobot/pick_lift_vla_images_baseline_v1",
    )
    parser.add_argument(
        "--repo-id", default="local/isaac-pick-lift-vla-images-baseline-v1"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def rgb_uint8(value) -> np.ndarray:
    image = to_numpy(value)
    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(np.rint(image * 255.0), 0, 255)
    return image.astype(np.uint8, copy=False)


def quaternion_error_deg(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted_norm = np.linalg.norm(predicted)
    target_norm = np.linalg.norm(target)
    if predicted_norm < 1e-8 or target_norm < 1e-8:
        return 180.0
    dot = abs(float(np.dot(predicted / predicted_norm, target / target_norm)))
    return float(np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0))))


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    rng = np.random.default_rng(args.seed)
    sample_count = min(args.samples, len(dataset))
    indices = np.sort(rng.choice(len(dataset), size=sample_count, replace=False))
    client = VLAClient(args.host, args.port, timeout_s=120.0)

    position_errors = []
    orientation_errors = []
    gripper_errors = []
    action_l2_errors = []
    latencies = []
    finite_chunks = 0

    started = time.perf_counter()
    for index in indices:
        sample = dataset[int(index)]
        request = VLARequest(
            rgb=rgb_uint8(sample["observation.images.table"]),
            state=to_numpy(sample["observation.state"]).astype(np.float32),
            task=str(sample.get("task", "Pick up the red cube and lift it 10 centimeters.")),
        )
        response = client.predict(request)
        predicted = response.actions[0]
        target = to_numpy(sample["action"]).astype(np.float32)

        finite_chunks += int(np.isfinite(response.actions).all())
        position_errors.append(float(np.linalg.norm(predicted[:3] - target[:3])))
        orientation_errors.append(quaternion_error_deg(predicted[3:7], target[3:7]))
        gripper_errors.append(float(abs(predicted[7] - target[7])))
        action_l2_errors.append(float(np.linalg.norm(predicted - target)))
        latencies.append(response.inference_ms)

    result = {
        "scope": "exported_dataset_imitation_sanity_check",
        "samples": sample_count,
        "dataset_frames": len(dataset),
        "finite_action_chunk_rate": finite_chunks / sample_count,
        "position_mae_cm": float(np.mean(position_errors) * 100.0),
        "position_p90_cm": float(np.percentile(position_errors, 90)),
        "orientation_mae_deg": float(np.mean(orientation_errors)),
        "gripper_mae": float(np.mean(gripper_errors)),
        "action_l2_mae": float(np.mean(action_l2_errors)),
        "inference_mean_ms": float(np.mean(latencies)),
        "inference_p90_ms": float(np.percentile(latencies, 90)),
        "wall_time_s": time.perf_counter() - started,
        "indices": indices.tolist(),
    }
    # Convert the p90 position metric from metres to centimetres as labeled.
    result["position_p90_cm"] *= 100.0
    print("VLA_ACTION_EVAL=" + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
