"""Optionally precompute features; the main inference CLI also accepts video directly."""

import argparse
from pathlib import Path

from worldsonus.checkpoint import atomic_save
from worldsonus.features import extract_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--dino", required=True)
    parser.add_argument("--text-encoder", required=True)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument(
        "--long-edge", type=int, help="Optional extra pre-resize; omit for reference preprocessing"
    )
    parser.add_argument("--frame-batch", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Output exists; choose a new file")
    kwargs = vars(args).copy()
    output = kwargs.pop("output")
    record = extract_features(**kwargs)
    atomic_save(record, output)
    print(f"Saved {record['video'].shape[0]} frames of features to {output}")


if __name__ == "__main__":
    main()
