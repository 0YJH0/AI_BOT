import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "full_export", Path(__file__).resolve().parents[1] / "scripts/export_vla_execution_aligned.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def metadata(initial_q):
    return {"domain": {"object": {}, "camera": {}, "lighting": {},
                       "collection_initial_joint_position": [initial_q]}}


def test_scene_group_uses_physical_parameters_not_sample_id():
    a = metadata(.1)
    b = metadata(.1) | {"sample_key": "different-id"}
    assert module.scene_group(a) == module.scene_group(b)
    assert module.scene_group(a) != module.scene_group(metadata(.2))


def test_split_is_episode_and_scene_disjoint(tmp_path, monkeypatch):
    mapping = {}
    for i, level in enumerate(module.LEVELS):
        (tmp_path / level).mkdir()
        for j in range(3):
            path = tmp_path / level / f"episode_{j:06d}.npz"
            path.touch()
            mapping[path] = metadata(i + j / 10)
    monkeypatch.setattr(module, "load_episode", lambda p: ({}, mapping[p]))
    splits = module.select_splits(tmp_path, 1, 1, 1, 42)
    groups = [{r[2] for r in records} for records in splits.values()]
    assert all(len(g) == 3 for g in groups)
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    mapping[next(iter(mapping))] = metadata(1.0)
    with pytest.raises(ValueError, match="Duplicate physical scene"):
        module.select_splits(tmp_path, 1, 1, 1, 42)
