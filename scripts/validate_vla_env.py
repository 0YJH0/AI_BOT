"""Validate the isolated LeRobot/SmolVLA environment without downloading weights."""

from __future__ import annotations

import json

import torch
import lerobot
import peft
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def main() -> None:
    del LeRobotDataset, make_pre_post_processors, SmolVLAPolicy, peft
    summary = {
        "status": "VLA_ENV_OK",
        "torch": torch.__version__,
        "lerobot": lerobot.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
        if torch.cuda.is_available()
        else 0.0,
    }
    print("VLA_ENV_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)
    if not summary["cuda_available"]:
        raise RuntimeError("CUDA is not available in lerobot-vla")


if __name__ == "__main__":
    main()
