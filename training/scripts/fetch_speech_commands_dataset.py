#!/usr/bin/env python3
"""Download and extract the TensorFlow Speech Commands v0.02 dataset."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from dataset_paths import DATASET_NAME, default_dataset_cache_dir

DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Speech Commands v0.02 to the local cache")
    p.add_argument(
        "--dataset-dir",
        default=str(default_dataset_cache_dir()),
        help="Extraction directory for the dataset root",
    )
    p.add_argument(
        "--url",
        default=DATASET_URL,
        help="Dataset archive URL",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Delete any existing dataset dir before extracting",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()

    if dataset_dir.exists():
        if not args.force:
            print(f"Dataset already present: {dataset_dir}")
            return 0
        shutil.rmtree(dataset_dir)

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="speech_commands_dl_") as tmpdir:
        archive_path = Path(tmpdir) / f"{DATASET_NAME}.tar.gz"
        print(f"Downloading {args.url}")
        print(f"Archive: {archive_path}")
        try:
            urllib.request.urlretrieve(args.url, archive_path)
        except Exception as exc:
            print(f"urllib download failed: {exc}")
            print("Falling back to curl -L --fail")
            subprocess.run(
                ["curl", "-L", "--fail", args.url, "-o", str(archive_path)],
                check=True,
            )

        extract_root = Path(tmpdir) / DATASET_NAME
        extract_root.mkdir(parents=True, exist_ok=True)

        print(f"Extracting to {extract_root}")
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(extract_root)

        shutil.move(str(extract_root), str(dataset_dir))

    print(f"Dataset ready at {dataset_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
