"""Decode representative training RGB frames from the embedded Parquet images."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grasp-lift", action="store_true",
                        help="Show late GRASP and LIFT frames from original training demonstrations.")
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite preview directory: {output}")
    manifest = json.loads((root / "balanced_split_manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    selections = []
    groups = ("grasp", "lift") if args.grasp_lift else ("original", "tight_recovery")
    for group in groups:
        for level in ("easy", "medium", "hard"):
            candidates = [
                i for i, record in enumerate(records)
                if record["level"] == level and (
                    not record.get("dagger_recovery", False)
                    if group != "tight_recovery"
                    else record.get("recovery_band_m", {}).get("minimum") == 0.01
                )
            ]
            if not candidates:
                raise ValueError(f"No matching episode for {group}/{level}")
            selections.append((group, level, candidates[0]))

    episode_ids = [item[2] for item in selections]
    columns = ["episode_index", "frame_index", "observation.images.table", "training.controller_phase"]
    rows = []
    for parquet in sorted((root / "data").glob("chunk-*/*.parquet")):
        rows.extend(pq.read_table(parquet, columns=columns,
                                  filters=[("episode_index", "in", episode_ids)]).to_pylist())

    output.mkdir(parents=True, exist_ok=True)
    margin, width, height, caption = 24, 320, 240, 58
    sheet = Image.new("RGB", (width * 3 + margin * 4, 720), "#f5f6f8")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
        heading_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
    except OSError:
        font = heading_font = ImageFont.load_default()
    draw.text((margin, 18), "RGB decoded from actual LeRobot training Parquet files", fill="#151b25", font=heading_font)
    previews = []
    for tile_index, (group, level, episode_id) in enumerate(selections):
        episode_rows = sorted((r for r in rows if r["episode_index"] == episode_id),
                              key=lambda r: r["frame_index"])
        if group == "tight_recovery":
            learner_frame = records[episode_id]["learner_visited_frame_indices"][0]
            row = next(r for r in episode_rows if r["frame_index"] == learner_frame)
        else:
            phase = {"grasp": 3, "lift": 4}.get(group, 1)
            matching = [r for r in episode_rows
                        if int(np.asarray(r["training.controller_phase"]).reshape(-1)[0]) == phase]
            if not matching and args.grasp_lift:
                raise ValueError(f"Episode {episode_id} has no phase {phase} frames")
            row = matching[-1] if matching else episode_rows[0]
        image_value = row["observation.images.table"]
        if not image_value.get("bytes"):
            raise ValueError("Expected embedded image bytes in this dataset")
        with Image.open(io.BytesIO(image_value["bytes"])) as decoded:
            rgb = decoded.convert("RGB")
            filename = f"{group}_{level}_ep{episode_id:06d}_frame{row['frame_index']:03d}.png"
            rgb.save(output / filename)
            pixels = np.asarray(rgb)
            if rgb.size != (width, height):
                raise ValueError(f"Expected native 320x240 images, got {rgb.size}")
            x = margin + (tile_index % 3) * (width + margin)
            y = 68 + (tile_index // 3) * (height + caption + margin)
            draw.text((x, y), f"{group.replace('_', ' ').title()} | {level.title()}", fill="#151b25", font=font)
            draw.text((x, y + 24), f"episode {episode_id}, frame {row['frame_index']}", fill="#536071", font=font)
            sheet.paste(rgb, (x, y + caption))
            previews.append({"group": group, "level": level,
                             "episode_index": episode_id, "frame_index": row["frame_index"],
                             "image": str(output / filename), "shape": list(pixels.shape),
                             "rgb_std": float(pixels.std()),
                             "source": records[episode_id].get("source")})
    contact_sheet = output / "training_rgb_preview.png"
    sheet.save(contact_sheet)
    (output / "preview_manifest.json").write_text(
        json.dumps({"dataset_root": str(root), "previews": previews}, indent=2), encoding="utf-8")
    print(json.dumps({"contact_sheet": str(contact_sheet), "previews": previews}, indent=2))


if __name__ == "__main__":
    main()
