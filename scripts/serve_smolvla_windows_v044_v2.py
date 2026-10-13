"""LeRobot 0.4.4 compatibility entry point for the SmolVLA RPC server.

LeRobot 0.4.4 keeps ``make_pre_post_processors`` in
``lerobot.policies.factory`` and does not re-export the SmolVLA class from the
package root. The shared server intentionally uses the newer public imports;
this small adapter supplies both symbols without changing the server itself.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import lerobot.policies as policies_package
import lerobot.policies.smolvla as smolvla_package
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


policies_package.make_pre_post_processors = make_pre_post_processors
smolvla_package.SmolVLAPolicy = SmolVLAPolicy

runpy.run_path(str(Path(__file__).with_name("serve_smolvla.py")), run_name="__main__")
