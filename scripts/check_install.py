#!/usr/bin/env python
"""Check the IDM-VTON demo environment without launching the UI."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


REQUIRED_MODULES = [
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "gradio",
    "onnxruntime",
    "cv2",
    "fvcore",
    "pycocotools",
    "av",
]

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
        "--skip-detectron2",
        action="store_true",
        help="Skip detectron2._C import check.",
    )
    return parser.parse_args()


def import_and_print(module_name: str) -> None:
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", "unknown")
    print(f"ok: {module_name} {version}")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    print(f"python: {sys.executable}")
    print(f"repo: {repo_root}")

    for module_name in REQUIRED_MODULES:
        import_and_print(module_name)

    import torch

    print(f"torch cuda: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
        print(f"capability: {torch.cuda.get_device_capability(0)}")

    for rel_path, min_size in CHECKPOINTS.items():
        target = repo_root / rel_path
        size = target.stat().st_size if target.exists() else 0
        if size < min_size:
            raise RuntimeError(f"missing or placeholder checkpoint: {rel_path} ({size} bytes)")
        print(f"ok: {rel_path} ({size} bytes)")

    if not args.skip_detectron2:
        sys.path.insert(0, str(repo_root / "gradio_demo"))
        import detectron2
        import detectron2._C as detectron2_ext
        import densepose

        print(f"ok: detectron2 {detectron2.__version__}")
        print(f"ok: detectron2._C {detectron2_ext.__file__}")
        print(f"ok: detectron2 compiler {detectron2_ext.get_compiler_version()}")
        print(f"ok: detectron2 cuda {detectron2_ext.get_cuda_version()}")
        print(f"ok: densepose {getattr(densepose, '__version__', 'unknown')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
