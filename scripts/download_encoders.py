"""Fetch pinned frozen encoders after accepting their upstream access terms."""

import argparse
from pathlib import Path

ENCODERS = {
    "dino": (
        "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        "c93d816fc9e567563bc068f01475bec89cc634a6",
    ),
    "text": ("google/t5gemma-2-270m-270m", "7c38f16641f455ef0685b18431faf1b17722d5a1"),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="assets")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    for name, (repo, revision) in ENCODERS.items():
        snapshot_download(
            repo_id=repo,
            revision=revision,
            local_dir=Path(args.output) / name,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"],
        )
        print(f"Downloaded {name} at revision {revision}")
