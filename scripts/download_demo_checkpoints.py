#!/usr/bin/env python
"""Download IDM-VTON Gradio preprocessing checkpoints into ./ckpt."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "yisol/IDM-VTON"
REPO_TYPE = "space"

CHECKPOINTS = {
    "ckpt/densepose/model_final_162be9.pkl": 10_000_000,
    "ckpt/humanparsing/parsing_atr.onnx": 10_000_000,
    "ckpt/humanparsing/parsing_lip.onnx": 10_000_000,
    "ckpt/openpose/ckpts/body_pose_model.pth": 10_000_000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="IDM-VTON repository root. Defaults to this checkout.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even if a non-placeholder file already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    for rel_path, min_size in CHECKPOINTS.items():
        target = repo_root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists() and target.stat().st_size >= min_size and not args.force:
            print(f"ok: {rel_path} ({target.stat().st_size} bytes)")
            continue

        print(f"download: {rel_path}")
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=rel_path,
            local_dir=repo_root,
            local_dir_use_symlinks=False,
            force_download=args.force,
        )

        size = target.stat().st_size if target.exists() else 0
        if size < min_size:
            raise RuntimeError(f"{rel_path} looks too small after download: {size} bytes")
        print(f"ok: {rel_path} ({size} bytes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
