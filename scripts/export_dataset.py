"""Export raw NPZ/JSON episodes into validated, sharded HDF5."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from isaac_lab_data_engine.data_collection.dataset_exporter import export_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", "--input-dir", dest="input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=Path, required=True)
    parser.add_argument("--format", choices=("hdf5",), default="hdf5")
    parser.add_argument(
        "--episodes_per_shard", "--episodes-per-shard", dest="episodes_per_shard",
        type=int, default=64,
    )
    parser.add_argument("--compression", choices=("gzip", "lzf", "none"), default="gzip")
    parser.add_argument("--compression_level", type=int, default=4)
    parser.add_argument(
        "--split_ratios", type=float, nargs=3, metavar=("TRAIN", "VALIDATION", "TEST"),
        default=(0.8, 0.1, 0.1),
    )
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume a matching staging export, or validate an already completed export.",
    )
    parser.add_argument(
        "--no_validate", action="store_true",
        help="Skip exact source-vs-HDF5 comparison (not recommended).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = export_dataset(
        args.input_dir,
        args.output_dir,
        episodes_per_shard=args.episodes_per_shard,
        compression=args.compression,
        compression_level=args.compression_level,
        split_ratios=tuple(args.split_ratios),
        split_seed=args.split_seed,
        resume=args.resume,
        validate=not args.no_validate,
    )
    print("PHASE6_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
