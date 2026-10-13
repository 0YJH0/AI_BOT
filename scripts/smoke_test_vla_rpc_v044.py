"""Send one real exported dataset sample through the local SmolVLA RPC server."""

from __future__ import annotations

import argparse
import json
import sys
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
        "--repo-id",
        default="local/isaac-pick-lift-vla-images-baseline-v1",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--sample-index", type=int, default=0)
    return parser.parse_args()


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def main() -> None:
    args = parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    sample = dataset[args.sample_index]

    image = _to_numpy(sample["observation.images.table"])
    if image.ndim != 3:
        raise ValueError(f"Expected one image, got {image.shape}")
    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(np.rint(image * 255.0), 0, 255)
    image = image.astype(np.uint8, copy=False)

    state = _to_numpy(sample["observation.state"]).astype(np.float32, copy=False)
    task = str(sample.get("task", "Pick up the red cube and lift it 10 centimeters."))

    response = VLAClient(args.host, args.port, timeout_s=120.0).predict(
        VLARequest(rgb=image, state=state, task=task)
    )
    if not np.isfinite(response.actions).all():
        raise ValueError("VLA returned non-finite actions")

    print(
        "VLA_RPC_SMOKE_OK="
        + json.dumps(
            {
                "sample_index": args.sample_index,
                "rgb_shape": list(image.shape),
                "state_shape": list(state.shape),
                "actions_shape": list(response.actions.shape),
                "inference_ms": round(response.inference_ms, 3),
                "first_action": response.actions[0].round(5).tolist(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
