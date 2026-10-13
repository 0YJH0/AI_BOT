import pytest
from isaac_lab_data_engine.vla.loss_plateau import LossPlateau, PlateauStop, install


def test_flat_requires_minimum_and_patience():
    m = LossPlateau(window=2, min_steps=12, patience=3)
    for _ in range(11):
        m.update(1.0)
        assert not m.stop
    assert m.update(1.0)['stop_requested']


def test_improvement_resets_and_small_gains_accumulate():
    m = LossPlateau(window=1, min_steps=4, patience=3)
    for v in [1.0, .995, .989, .988]:
        m.update(v)
    assert m.stale == 1 and not m.stop
    with pytest.raises(FloatingPointError):
        m.update(float('nan'))


def test_stop_only_after_checkpoint(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    monkeypatch.setenv('LEROBOT_LOSS_PLATEAU_JSON', json.dumps(dict(
        output=str(tmp_path), window=1, min_steps=2, patience=1, relative=.01)))
    saved = []
    module = SimpleNamespace(update_policy=lambda: (SimpleNamespace(loss=SimpleNamespace(val=1.0)), {}),
                             update_last_checkpoint=lambda p: saved.append(p))
    install(module)
    module.update_policy()
    module.update_policy()
    with pytest.raises(PlateauStop):
        module.update_last_checkpoint(tmp_path/'000002')
    assert saved == [tmp_path/'000002']
    assert json.loads((tmp_path/'training_progress.json').read_text())['status'] == 'EARLY_STOPPED'
