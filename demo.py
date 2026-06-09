#!/usr/bin/env python
"""Launch the IDM-VTON Gradio demo with explicit local settings."""

from __future__ import annotations

import argparse
import os
import runpy
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--model", default="yisol/IDM-VTON")
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    os.environ["IDM_VTON_HOST"] = args.host
    os.environ["IDM_VTON_PORT"] = str(args.port)
    os.environ["IDM_VTON_SHARE"] = "1" if args.share else "0"
    os.environ["IDM_VTON_MODEL"] = args.model
    if args.device:
        os.environ["IDM_VTON_DEVICE"] = args.device

    runpy.run_path(str(repo_root / "gradio_demo" / "app.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
