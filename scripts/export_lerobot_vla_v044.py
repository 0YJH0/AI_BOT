"""LeRobot 0.4.4 compatibility entry point for export_lerobot_vla.py."""

from __future__ import annotations

import runpy
from pathlib import Path

import lerobot.datasets as datasets_package
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# LeRobot 0.6 exports this class at package level; 0.4.4 does not.  Injecting
# the same public name keeps the version-independent exporter as the source of
# truth and avoids maintaining a second conversion implementation.
datasets_package.LeRobotDataset = LeRobotDataset
runpy.run_path(str(Path(__file__).with_name("export_lerobot_vla.py")), run_name="__main__")
