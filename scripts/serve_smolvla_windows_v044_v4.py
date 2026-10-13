"""Verified LeRobot 0.4.4 / Windows SmolVLA RPC entry point.

It adapts the shared server to LeRobot 0.4.4's factory exports and loads a PEFT
checkpoint as base SmolVLA weights plus the saved LoRA adapter.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import lerobot.policies as policies_package
import lerobot.policies.smolvla as smolvla_package
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaac_lab_data_engine.vla.checkpoint import load_lerobot_policy_config


_original_from_pretrained = SmolVLAPolicy.from_pretrained


@classmethod
def _from_pretrained_with_peft(cls, pretrained_name_or_path, *args, **kwargs):
    checkpoint = Path(pretrained_name_or_path)
    adapter_config_path = checkpoint / "adapter_config.json"
    if not adapter_config_path.is_file():
        return _original_from_pretrained(pretrained_name_or_path, *args, **kwargs)

    from peft import PeftConfig, PeftModel

    # Load through the union base class so the serialized `type: smolvla` field
    # selects SmolVLAConfig before CLI-style overrides are applied.
    policy_config = load_lerobot_policy_config(checkpoint)
    peft_config = PeftConfig.from_pretrained(str(checkpoint))
    base_path = peft_config.base_model_name_or_path
    if not base_path:
        metadata = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_path = metadata.get("base_model_name_or_path")
    if not base_path:
        raise ValueError("PEFT checkpoint does not name its SmolVLA base model")

    base_policy = _original_from_pretrained(base_path, config=policy_config)
    policy = PeftModel.from_pretrained(
        base_policy,
        str(checkpoint),
        config=peft_config,
    )
    policy.config.use_peft = True
    return policy


policies_package.make_pre_post_processors = make_pre_post_processors
SmolVLAPolicy.from_pretrained = _from_pretrained_with_peft
smolvla_package.SmolVLAPolicy = SmolVLAPolicy

runpy.run_path(str(Path(__file__).with_name("serve_smolvla.py")), run_name="__main__")
