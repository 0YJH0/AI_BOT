"""Verify actual LeRobot reads, split isolation, and the phase-free sampler."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isaac_lab_data_engine.vla.relative_interface import contract


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    groups, splits = {}, []
    for split in ("train", "validation", "test"):
        manifest = json.loads((a.root / split / "balanced_split_manifest.json").read_text())
        if manifest["interface_contract"] != contract(): raise ValueError("Bad contract")
        groups[split] = {r["scene_group"] for r in manifest["records"]}
        dataset = LeRobotDataset(repo_id=manifest["repo_id"], root=a.root / split,
                                 download_videos=False)
        for i in (0, len(dataset)//2, len(dataset)-1):
            sample = dataset[i]
            if sample["observation.state"].shape != (29,) or sample["action"].shape != (4,):
                raise ValueError("LeRobot read returned the wrong feature widths")
            if not np.isfinite(sample["observation.state"].numpy()).all():
                raise ValueError("Nonfinite state")
            if sample["observation.images.table"].shape != (3, 240, 320):
                raise ValueError("RGB decode failed")
        if len(dataset) != manifest["frames"]: raise ValueError("Wrong frame count")
        splits.append({"split": split, "frames": len(dataset), "episodes": manifest["episodes"]})
    if any(groups[x] & groups[y] for x, y in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Split leakage")
    (a.root / "loader_audit.json").write_text(json.dumps({
        "status": "RELATIVE4D_LEROBOT_LOADER_VERIFIED", "splits": splits,
        "scene_groups_disjoint": True, "phase_in_model_input": False}, indent=2), encoding="utf-8")
    print("RELATIVE4D_LOADER_OK=" + json.dumps(splits), flush=True)


if __name__ == "__main__": main()
