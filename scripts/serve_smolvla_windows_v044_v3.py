"""Windows/LeRobot 0.4.4 SmolVLA RPC entry point with PEFT loading.

This adapter keeps the shared RPC server unchanged while supplying the API
exports and PEFT loading behavior required by LeRobot 0.4.4 checkpoints.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import lerobot.policies as policies_package
import lerobot.policies.smolvla as smolvla_package
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


_original_from_pretrained = SmolVLAPolicy.from_pretrained


@classmethod
def _from_pretrained_with_peft(cls, pretrained_name_or_path, *args, **kwargs):
    checkpoint = Path(pretrained_name_or_path)
    adapter_config_path = checkpoint / "adapter_config.json"
    if not adapter_config_path.is_file():
        return _original_from_pretrained(pretrained_name_or_path, *args, **kwargs)

    from peft import PeftConfig, PeftModel

    policy_config = SmolVLAConfig.from_pretrained(str(checkpoint))
    peft_config = PeftConfig.from_pretrained(str(checkpoint))
    base_path = peft_config.base_model_name_or_path
    if not base_path:
        # Retain a readable fallback for adapters moved between machines.
        metadata = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_path = metadata.get("base_model_name_or_path")
    if not base_path:
        raise ValueError("PEFT checkpoint does not name its SmolVLA base model")

    base_policy = _original_from_pretrained(
        base_path,
        config=policy_config,
    )
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
