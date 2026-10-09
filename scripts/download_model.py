"""Download the inference bundle from Hugging Face."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ASSETS = ("worldsonus_150k_padtrim.pt", "audio_codec.pt", "z_stats.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="FF2416/WorldSonus")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, default=Path("assets"))
    args = parser.parse_args()
    # Pin the branch once so a concurrent publication cannot mix versions.
    revision = HfApi().model_info(args.repo, revision=args.revision).sha
    for name in ASSETS:
        hf_hub_download(args.repo, name, revision=revision, local_dir=args.output)
        print(f"Downloaded {name}")
    print(f"Model revision: {revision}")


if __name__ == "__main__":
    main()
