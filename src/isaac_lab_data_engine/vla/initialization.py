"""Audit a newly created LoRA before its first optimizer update."""
import hashlib
import numpy as np


def audit_fresh_lora(model, *, expected_parameters=742656):
    configs = model.peft_config
    if sorted(configs) != ["default"]:
        raise ValueError("Expected exactly one fresh default adapter")
    config = configs["default"]
    if config.r != 16 or config.lora_alpha != 8 or config.lora_dropout != 0:
        raise ValueError("Fresh LoRA hyperparameters must match the continued arm")
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if sum(p.numel() for _, p in parameters) != expected_parameters:
        raise ValueError("Unexpected trainable parameter count")
    if any("lora_A" not in name and "lora_B" not in name for name, _ in parameters):
        raise ValueError("Base parameters must remain frozen")
    b_parameters = [(n, p) for n, p in parameters if "lora_B" in n]
    if not b_parameters or any(p.detach().count_nonzero().item() for _, p in b_parameters):
        raise ValueError("Fresh LoRA B matrices must all be zero before training")
    digest = hashlib.sha256()
    for name, p in sorted(parameters):
        value = p.detach().cpu().float().contiguous().numpy()
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite initialization")
        digest.update(name.encode())
        digest.update(value.tobytes())
    return {
        "status": "FRESH_LORA_INITIALIZATION_VERIFIED", "old_adapter_loaded": False,
        "base_weights_frozen": True, "adapter_names": ["default"],
        "trainable_parameters": expected_parameters, "trainable_tensors": len(parameters),
        "zero_initialized_B_matrices": len(b_parameters), "initial_adapter_sha256": digest.hexdigest(),
        "r": config.r, "lora_alpha": config.lora_alpha, "lora_dropout": config.lora_dropout,
        "target_modules": config.target_modules,
    }
