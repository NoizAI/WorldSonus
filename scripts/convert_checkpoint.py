"""Export the final model without optimizer state or private training metadata."""

import argparse
from pathlib import Path
from worldsonus.checkpoint import export_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--trust-input-pickle",
        action="store_true",
        help="Only for trusted original training checkpoints with argparse metadata",
    )
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Output already exists; choose a new file")
    kept, dropped = export_checkpoint(
        args.input, args.output, trusted_pickle=args.trust_input_pickle
    )
    print(f"Exported {kept} tensors; removed {dropped} inactive/training-only tensors")
