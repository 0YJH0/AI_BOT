import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def audit(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('v9_audit', scripts/'audit_vla_v9_regression.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lock_detects_changed_or_missing_artifacts(audit, tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    f = tmp_path/'config.json'
    f.write_text('{}')
    lock = {'files': {'config.json': audit.digest(f)}}
    audit.verify_lock(lock)
    f.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='Frozen artifacts changed'):
        audit.verify_lock(lock)
    with pytest.raises(ValueError, match='Frozen artifacts changed'):
        audit.verify_lock({'files': {'missing': 'abc'}})


def test_scene_comparison_rejects_missing_or_changed_scene(audit):
    row = {k: 'same' for k in audit.SCENE_FIELDS}
    a = {'easy:1': row}
    audit.verify_scenes(a, {'easy:1': row.copy()})
    with pytest.raises(ValueError, match='Scene keys differ'):
        audit.verify_scenes(a, {})
    with pytest.raises(ValueError, match='Scene changed'):
        audit.verify_scenes(a, {'easy:1': {**row, 'object_x_m': 'changed'}})
