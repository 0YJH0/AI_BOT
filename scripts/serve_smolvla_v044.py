"""LeRobot 0.4.4 compatibility entry point for serve_smolvla.py."""

from __future__ import annotations

import runpy
from pathlib import Path

import lerobot.policies.smolvla as smolvla_package
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


smolvla_package.SmolVLAPolicy = SmolVLAPolicy
runpy.run_path(str(Path(__file__).with_name("serve_smolvla.py")), run_name="__main__")
