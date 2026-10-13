"""Windows-compatible entry point for LeRobot training.

LeRobot 0.4.4 creates ``checkpoints/last`` as a symbolic link. Creating that
link requires elevated privileges on a default Windows install. This launcher
replaces only that helper with a directory junction, keeping LeRobot's resume
path and checkpoint layout unchanged.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import wraps
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaac_lab_data_engine.vla.checkpoint import ensure_policy_config_file


def _prepare_local_policy_config(argv: list[str]) -> Path | None:
    """Restore metadata for local adapter-only inputs before LeRobot parses CLI."""
    policy_path = None
    for index, argument in enumerate(argv):
        if argument.startswith("--policy.path="):
            policy_path = argument.split("=", 1)[1]
            break
        if argument == "--policy.path" and index + 1 < len(argv):
            policy_path = argv[index + 1]
            break
    if policy_path is None:
        return None
    checkpoint = Path(policy_path).expanduser()
    if not checkpoint.is_dir():
        return None  # Leave Hub identifiers and normal CLI validation to LeRobot.
    if ensure_policy_config_file(checkpoint):
        print(f"Materialized input PEFT policy config: {checkpoint / 'config.json'}", flush=True)
    return checkpoint


def _update_last_checkpoint_windows(checkpoint_dir: str | Path) -> None:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    last_dir = checkpoint_dir.parent / "last"

    pretrained_model_dir = checkpoint_dir / "pretrained_model"
    if pretrained_model_dir.is_dir() and ensure_policy_config_file(
        pretrained_model_dir
    ):
        print(f"Materialized PEFT policy config: {pretrained_model_dir / 'config.json'}")

    if last_dir.exists() or os.path.lexists(last_dir):
        # rmdir removes a junction itself, not its target. A real non-empty
        # directory fails safely instead of deleting checkpoint contents.
        remove = subprocess.run(
            ["cmd", "/d", "/c", "rmdir", str(last_dir)],
            capture_output=True,
            text=True,
            encoding="utf-8", errors="replace",
            check=False,
        )
        if remove.returncode != 0:
            raise RuntimeError(
                f"Could not replace checkpoint marker {last_dir}: "
                f"{remove.stderr or remove.stdout}"
            )

    create = subprocess.run(
        ["cmd", "/d", "/c", "mklink", "/J", str(last_dir), str(checkpoint_dir)],
        capture_output=True,
        text=True,
        encoding="utf-8", errors="replace",
        check=False,
    )
    if create.returncode != 0:
        raise RuntimeError(
            f"Could not create checkpoint junction {last_dir}: "
            f"{create.stderr or create.stdout}"
        )

    print(f"Windows checkpoint marker: {last_dir} -> {checkpoint_dir.name}")
    contract_path = os.environ.get("LEROBOT_INTERFACE_CONTRACT")
    if contract_path:
        payload = json.loads(Path(contract_path).read_text(encoding="utf-8"))
        from isaac_lab_data_engine.vla.relative_interface import contract
        if payload != contract():
            raise ValueError("Invalid interface contract")
        cfg = json.loads((pretrained_model_dir / "config.json").read_text(encoding="utf-8"))
        if (cfg["input_features"]["observation.state"]["shape"] != [29]
                or cfg["output_features"]["action"]["shape"] != [4]):
            raise ValueError("Checkpoint features do not match the relative interface")
        (pretrained_model_dir / "interface_contract.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
    joint_contract_path = os.environ.get("LEROBOT_JOINT_INTERFACE_CONTRACT")
    if joint_contract_path:
        if contract_path:
            raise ValueError("Cannot mix relative and joint contracts")
        from isaac_lab_data_engine.vla.joint_interface import contract
        payload = json.loads(Path(joint_contract_path).read_text(encoding="utf-8-sig"))
        cfg = json.loads((pretrained_model_dir / "config.json").read_text(encoding="utf-8"))
        if (payload != contract() or cfg["input_features"]["observation.state"]["shape"] != [25]
                or cfg["output_features"]["action"]["shape"] != [8]):
            raise ValueError("Joint checkpoint must use robot-only 25 / joint 8 features")
        (pretrained_model_dir / "interface_contract.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Environment-installed objectives are not serialized by upstream train_config.
    # Keep them alongside each checkpoint so a later continuation is auditable.
    loss_config = os.environ.get("LEROBOT_SMOLVLA_WEIGHTED_LOSS_JSON")
    if loss_config:
        import hashlib
        plan = os.environ.get("LEROBOT_VLA_SAMPLING_PLAN")
        objective = {"weighted_flow_matching": json.loads(loss_config), "sampling_plan": plan}
        if plan:
            objective["sampling_plan_sha256"] = hashlib.sha256(Path(plan).read_bytes()).hexdigest()
        (pretrained_model_dir / "training_objective.json").write_text(
            json.dumps(objective, indent=2), encoding="utf-8")


def _install_smolvla_padding_compatibility() -> None:
    """Bridge LeRobot 0.4.4's ``action_is_pad``/``actions_id_pad`` typo.

    The dataset emits ``action_is_pad`` while SmolVLA 0.4.4 looks up the
    pluralized ``actions_id_pad`` key.  Aliasing the tensor at the policy
    boundary keeps padded action-chunk targets out of the flow-matching loss
    without modifying the installed package.
    """

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    original_forward = SmolVLAPolicy.forward

    @wraps(original_forward)
    def forward_with_padding_alias(self, batch, *args, **kwargs):
        if "action_is_pad" in batch and "actions_id_pad" not in batch:
            batch = dict(batch)
            batch["actions_id_pad"] = batch["action_is_pad"]
        return original_forward(self, batch, *args, **kwargs)

    SmolVLAPolicy.forward = forward_with_padding_alias
    print("SmolVLA padding compatibility: action_is_pad -> actions_id_pad")


def _install_environment_cli_overrides() -> None:
    """Inject structured CLI values without PowerShell stripping JSON quotes."""

    transforms_json = os.environ.get("LEROBOT_IMAGE_TRANSFORMS_TFS_JSON")
    if not transforms_json:
        return

    try:
        normalized = json.dumps(json.loads(transforms_json), separators=(",", ":"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "LEROBOT_IMAGE_TRANSFORMS_TFS_JSON is not valid JSON"
        ) from exc

    argument_prefix = "--dataset.image_transforms.tfs="
    sys.argv = [arg for arg in sys.argv if not arg.startswith(argument_prefix)]
    sys.argv.append(argument_prefix + normalized)
    print("LeRobot CLI compatibility: injected image transforms from environment")


def _install_weighted_smolvla_loss_from_environment() -> None:
    """Install the opt-in real-dimension/Z/sample-weighted SmolVLA loss."""

    config_json = os.environ.get("LEROBOT_SMOLVLA_WEIGHTED_LOSS_JSON")
    if not config_json:
        return
    try:
        payload = json.loads(config_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "LEROBOT_SMOLVLA_WEIGHTED_LOSS_JSON is not valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("SmolVLA weighted-loss configuration must be a JSON object")
    allowed = {"z_axis_index", "z_axis_weight", "sample_weight_key", "sample_weight_reduction",
               "gripper_index", "gripper_weight", "lift_z_axis_weight"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise RuntimeError(
            "Unknown SmolVLA weighted-loss settings: " + ", ".join(unknown)
        )

    from isaac_lab_data_engine.vla.weighted_smolvla_loss import (
        WeightedSmolVLALossConfig,
        install_weighted_smolvla_loss,
    )

    config = WeightedSmolVLALossConfig(**payload)
    install_weighted_smolvla_loss(config)
    print(
        "SmolVLA weighted loss: "
        f"real action dims only, z[{config.z_axis_index}]={config.z_axis_weight}, "
        f"sample key={config.sample_weight_key}, reduction={config.sample_weight_reduction}, "
        f"gripper[{config.gripper_index}]={config.gripper_weight}"
    )


def _install_weighted_sampler_from_environment() -> None:
    """Make LeRobot's offline DataLoader consume a verified sampling plan."""

    plan_value = os.environ.get("LEROBOT_VLA_SAMPLING_PLAN")
    if not plan_value:
        return

    import torch

    from isaac_lab_data_engine.vla.sampling import (
        load_sampling_plan,
        make_weighted_random_sampler,
    )

    plan_path = Path(plan_value).expanduser().resolve()
    try:
        plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read VLA sampling plan {plan_path}") from exc
    dataset_root_value = plan_payload.get("dataset_root")
    if not dataset_root_value:
        raise RuntimeError(f"Sampling plan has no dataset_root: {plan_path}")
    planned_dataset_root = Path(str(dataset_root_value)).expanduser().resolve()
    original_dataloader = torch.utils.data.DataLoader

    class SamplingPlanDataLoader(original_dataloader):
        _isaac_sampling_plan = str(plan_path)

        def __init__(self, dataset, *args, **kwargs):
            dataset_root = getattr(dataset, "root", None)
            resolved_dataset_root = (
                Path(dataset_root).expanduser().resolve()
                if dataset_root is not None
                else None
            )
            if resolved_dataset_root == planned_dataset_root:
                manifest_path = planned_dataset_root / "balanced_split_manifest.json"
                weights, verified = load_sampling_plan(
                    plan_path,
                    dataset_length=len(dataset),
                    dataset_manifest_path=manifest_path,
                )
                sampler = make_weighted_random_sampler(
                    weights,
                    num_samples=int(verified["num_samples"]),
                    seed=int(verified["seed"]),
                )
                kwargs["shuffle"] = False
                kwargs["sampler"] = sampler
                print(
                    "VLA weighted sampler: "
                    f"frames={len(weights)}, draws={verified['num_samples']}, "
                    f"seed={verified['seed']}, plan={plan_path}"
                )
                if os.environ.get("LEROBOT_LIFT_PHASE_METADATA") == "1":
                    from isaac_lab_data_engine.vla.phase_loss_dataset import PhaseLossDataset
                    dataset = PhaseLossDataset(dataset)
            super().__init__(dataset, *args, **kwargs)

    torch.utils.data.DataLoader = SamplingPlanDataLoader
    print(f"VLA sampling plan armed for dataset: {planned_dataset_root}")


def _install_trainable_existing_peft_from_environment() -> None:
    """Load one existing LoRA adapter as trainable for fresh-optimizer tuning.

    PEFT defaults ``from_pretrained`` to inference-only. LeRobot 0.4.4 does
    not override that when ``--policy.path`` names an adapter checkpoint, so
    an apparently valid run would otherwise contain zero trainable params.
    This compatibility path is explicit and does not affect fresh LoRA runs.
    """

    enabled = os.environ.get("LEROBOT_TRAIN_EXISTING_PEFT", "").strip().lower()
    if enabled not in {"1", "true", "yes"}:
        return

    from peft import PeftModel

    original_from_pretrained = PeftModel.from_pretrained.__func__

    @classmethod
    def trainable_from_pretrained(cls, *args, **kwargs):
        kwargs["is_trainable"] = True
        model = original_from_pretrained(cls, *args, **kwargs)
        adapter_names = sorted(str(name) for name in model.peft_config)
        if adapter_names != ["default"]:
            raise RuntimeError(
                "Expected exactly one PEFT adapter named 'default', got "
                f"{adapter_names}"
            )
        model.set_adapter("default")
        trainable = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        trainable_tensors = sum(
            parameter.requires_grad for parameter in model.parameters()
        )
        if trainable <= 0:
            raise RuntimeError("Existing PEFT adapter has zero trainable parameters")
        expected_value = os.environ.get("LEROBOT_EXPECTED_PEFT_TRAINABLE_PARAMS")
        if expected_value and trainable != int(expected_value):
            raise RuntimeError(
                "Existing PEFT trainable-parameter mismatch: "
                f"expected={expected_value}, actual={trainable}"
            )
        print(
            "Existing PEFT adapter enabled: "
            f"adapters={adapter_names}, trainable_params={trainable}, "
            f"trainable_tensors={trainable_tensors}"
        )
        audit_path = os.environ.get("LEROBOT_CHUNK_ABLATION_AUDIT")
        if audit_path:
            from isaac_lab_data_engine.vla.chunk_ablation import audit_initialization
            audit_initialization(model, Path(audit_path), int(os.environ["LEROBOT_CHUNK_ABLATION_EXPECTED_CHUNK"]))
        return model

    PeftModel.from_pretrained = trainable_from_pretrained
    print("PEFT compatibility: existing adapter will load with is_trainable=True")


def _install_fresh_lora_audit_from_environment() -> None:
    output = os.environ.get("LEROBOT_FRESH_LORA_AUDIT_PATH")
    if not output:
        return
    if os.environ.get("LEROBOT_TRAIN_EXISTING_PEFT"):
        raise ValueError("Cannot combine fresh LoRA and existing-adapter training")
    from lerobot.policies.pretrained import PreTrainedPolicy
    from isaac_lab_data_engine.vla.initialization import audit_fresh_lora
    original = PreTrainedPolicy.wrap_with_peft

    @wraps(original)
    def audited_wrap(self, *args, **kwargs):
        if Path(self.config.pretrained_path).joinpath("adapter_config.json").exists():
            raise ValueError("Fresh LoRA cannot load previous adapter weights")
        model = original(self, *args, **kwargs)
        report = audit_fresh_lora(model)
        report["base_model"] = str(self.config.pretrained_path)
        path = Path(output)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("FRESH_LORA_INITIALIZATION=" + json.dumps(report), flush=True)
        return model

    PreTrainedPolicy.wrap_with_peft = audited_wrap


def main() -> None:
    checkpoint = _prepare_local_policy_config(sys.argv[1:])
    if "--checkpoint-preflight-only" in sys.argv:
        if checkpoint is None:
            raise ValueError("Checkpoint preflight requires a local --policy.path directory")
        # Policy subclasses register themselves on import; the normal trainer
        # imports the factory, but this lightweight preflight skips that path.
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
        from isaac_lab_data_engine.vla.checkpoint import load_lerobot_policy_config

        config = load_lerobot_policy_config(checkpoint)
        print(f"CHECKPOINT_PREFLIGHT_OK: type={config.type}, chunk_size={config.chunk_size}", flush=True)
        return
    from lerobot.utils import train_utils
    from lerobot.scripts import lerobot_train

    # LeRobot imports the helper into its training module namespace, so both
    # references must be updated before the CLI entry function starts.
    train_utils.update_last_checkpoint = _update_last_checkpoint_windows
    lerobot_train.update_last_checkpoint = _update_last_checkpoint_windows
    _install_smolvla_padding_compatibility()
    _install_weighted_smolvla_loss_from_environment()
    _install_weighted_sampler_from_environment()
    _install_trainable_existing_peft_from_environment()
    _install_fresh_lora_audit_from_environment()
    _install_environment_cli_overrides()
    from isaac_lab_data_engine.vla.loss_plateau import install, PlateauStop
    install(lerobot_train)
    try:
        lerobot_train.main()
    except PlateauStop as exc:
        print(f"Training stopped normally after loss plateau; saved checkpoint: {exc}", flush=True)


if __name__ == "__main__":
    main()
