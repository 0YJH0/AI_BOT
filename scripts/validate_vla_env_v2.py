"""Validate LeRobot 0.4.4, SmolVLA imports, CUDA, and the local GPU."""

from __future__ import annotations

import json

import lerobot
import peft  # noqa: F401
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
from lerobot.policies.factory import make_pre_post_processors  # noqa: F401
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: F401


def main() -> None:
    summary = {
        "status": "VLA_ENV_OK",
        "torch": torch.__version__,
        "lerobot": lerobot.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
        if torch.cuda.is_available()
        else 0.0,
        "smolvla_import": True,
        "dataset_v3_import": True,
    }
    print("VLA_ENV_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)
    if not summary["cuda_available"]:
        raise RuntimeError("CUDA is not available in lerobot-vla")


if __name__ == "__main__":
    main()
