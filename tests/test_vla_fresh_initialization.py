from types import SimpleNamespace
import pytest
import torch
from isaac_lab_data_engine.vla.initialization import audit_fresh_lora


class FakeFresh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.ones(3), requires_grad=False)
        self.lora_A = torch.nn.Parameter(torch.ones(2))
        self.lora_B = torch.nn.Parameter(torch.zeros(2))
        self.peft_config = {"default": SimpleNamespace(r=16, lora_alpha=8, lora_dropout=0, target_modules="test")}


def test_fresh_zero_adapter_is_auditable():
    report = audit_fresh_lora(FakeFresh(), expected_parameters=4)
    assert report["old_adapter_loaded"] is False
    assert report["zero_initialized_B_matrices"] == 1
    assert report["trainable_parameters"] == 4


@pytest.mark.parametrize("bad", ["old_b", "base", "rank", "stacked", "count"])
def test_rejects_old_weights_or_changed_training_scope(bad):
    model = FakeFresh()
    expected = 4
    if bad == "old_b": model.lora_B.data[0] = .01
    if bad == "base": model.base.requires_grad_(True); expected = 7
    if bad == "rank": model.peft_config["default"].r = 8
    if bad == "stacked": model.peft_config["other"] = model.peft_config["default"]
    if bad == "count": expected = 5
    with pytest.raises(ValueError): audit_fresh_lora(model, expected_parameters=expected)
