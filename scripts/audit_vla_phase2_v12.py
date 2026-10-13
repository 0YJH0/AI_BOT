"""Read-only label/normalizer audit and observable-phase replay on held-out demos."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pyarrow.parquet as pq
from safetensors.numpy import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isaac_lab_data_engine.vla.phase_diagnostics import FeedbackPhase


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def describe(x):
    return {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
            "min": x.min(0).tolist(), "max": x.max(0).tolist()}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    pre = load_file(str(a.checkpoint / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
    post = load_file(str(a.checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    train_stats = read(a.dataset / "train/meta/stats.json")
    normalization = {}
    for key in ("observation.state", "action"):
        normalization[key] = {}
        for kind in ("mean", "std"):
            actual = pre[key + "." + kind]
            expected = np.asarray(train_stats[key][kind])
            normalization[key][kind + "_train_stats_max_error"] = float(np.max(abs(actual - expected)))
    normalization["pre_post_action_stats_max_error"] = max(
        float(abs(pre["action." + k] - post["action." + k]).max()) for k in ("mean", "std"))
    phases = {}
    decoded_error = actual_applied_error = 0.
    all_state, all_action = [], []
    manifest = read(a.dataset / "train/balanced_split_manifest.json")
    for path in sorted((a.dataset / "train/data").glob("*/*.parquet")):
        t = pq.read_table(path, columns=["observation.state", "action"])
        all_state.extend(t["observation.state"].to_pylist())
        all_action.extend(t["action"].to_pylist())
    state, action = np.asarray(all_state, np.float32), np.asarray(all_action, np.float32)
    norm = (action - pre["action.mean"]) / (pre["action.std"] + 1e-8)
    restored = norm * (post["action.std"] + 1e-8) + post["action.mean"]
    decoded_error = float(abs(restored - action).max())
    for phase in range(5):
        mask = state[:, 29] == phase
        x = action[mask]
        visual_relative = state[mask, 25:28] - state[mask, 18:21]
        phases[str(phase)] = {"frames": int(mask.sum()), "action_statistics": describe(x),
            "visual_relative_xyz_statistics": describe(visual_relative),
            "closed_fraction": float((x[:, 7] < 0).mean()),
            "negative_z_fraction": float((x[:, 2] < -.001).mean()),
            "positive_z_fraction": float((x[:, 2] > .001).mean()),
            "normalized_xyz_statistics": describe(norm[mask, :3])}
    # Raw labels are compared with the executed command, not regenerated from
    # a new teacher. Only small numeric arrays are read; no image decode.
    for record in manifest["records"]:
        with np.load(record["source"], allow_pickle=False) as z:
            decoded = z["action"].copy()
            decoded[:, :3] += z["state"][:, 18:21]
            actual_applied_error = max(actual_applied_error, float(abs(decoded-z["applied_absolute_action"]).max()))
    # Replay the exact legacy FSM and the new feedback FSM on validation
    # observation histories, keeping all recorded actions completely unchanged.
    import torch
    from isaac_lab_data_engine.controllers.scripted_pick_lift import PickAndLiftStateMachine
    val = read(a.dataset / "validation/balanced_split_manifest.json")
    replay = []
    for record in val["records"]:
        with np.load(record["source"], allow_pickle=False) as z:
            s, steps, eef, q, dq, applied = (z[k] for k in (
                "state", "control_step", "physics_eef", "physics_q", "physics_dq", "physics_action"))
        visual = s[0, 25:28]
        legacy, feedback = PickAndLiftStateMachine(.02, 1, "cpu"), FeedbackPhase(.02)
        obj = torch.tensor([[*visual, 1., 0., 0., 0.]], dtype=torch.float32)
        lift = obj.clone(); lift[:, 2] += .18
        lookup = {int(k): i for i, k in enumerate(steps)}
        l, f, teacher = [], [], []
        latched = False
        for k in range(1, int(steps[-1]) + 1):
            if k in lookup:
                i = lookup[k]
                l.append(int(legacy.state[0])); f.append(feedback.phase); teacher.append(int(s[i, 29]))
            legacy.compute(torch.tensor(eef[k:k+1]), obj, lift)
            feedback.observe(eef[k, :3], visual, q[k, -2:], dq[k, -2:], latched)
            latched = latched or bool(applied[k, 7] < 0 and np.linalg.norm(eef[k, :3]-visual) <= .075)
        teacher, l, f = (np.asarray(x) for x in (teacher, l, f))
        replay.append({"sample_key": record["sample_key"], "level": record["level"], "frames": len(teacher),
            "legacy_mismatch_frames": int((l != teacher).sum()),
            "feedback_mismatch_frames": int((f != teacher).sum()),
            "phase2_frames": int((teacher == 2).sum()),
            "legacy_phase2_mismatch_frames": int(((teacher == 2) & (l != teacher)).sum()),
            "feedback_phase2_mismatch_frames": int(((teacher == 2) & (f != teacher)).sum()),
            "teacher_final_phase": int(teacher[-1]), "legacy_final_phase": int(l[-1]), "feedback_final_phase": int(f[-1])})
    counts = {k: sum(e[k] for e in replay) for k in (
        "frames", "legacy_mismatch_frames", "feedback_mismatch_frames", "phase2_frames",
        "legacy_phase2_mismatch_frames", "feedback_phase2_mismatch_frames")}
    result = {"status": "PHASE2_LABEL_NORMALIZATION_AUDIT_COMPLETE", "training_frames": len(action),
        "normalization": normalization, "normalization_roundtrip_max_error": decoded_error,
        "actual_executed_label_max_error": actual_applied_error, "phase_statistics": phases,
        "validation_observable_phase_replay": {"counts": counts, "episodes": replay},
        "cautions": ["Replay evaluates phases on successful expert histories, not student rollouts.",
                     "Truth teacher phase is an input discrepancy; no action relabeling was performed."]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("PHASE2_AUDIT=" + json.dumps({"output": str(a.output), "normalization": normalization,
         "counts": counts, "phase2": phases["2"]}), flush=True)


if __name__ == "__main__":
    main()
