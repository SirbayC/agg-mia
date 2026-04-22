#!/usr/bin/env python3
"""Preflight checks for the DelftBlue runtime environment."""

from __future__ import annotations

import platform
import sys


def main() -> int:
    print("DelftBlue Runtime Environment Check")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")

    try:
        import torch
    except Exception as exc:  # pragma: no cover - runtime preflight only
        print(f"Torch import failed: {exc}")
        return 1

    print(f"Torch: {torch.__version__}")
    print(f"Torch CUDA: {torch.version.cuda}")
    print(f"Torch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")

    try:
        import torchvision
    except Exception as exc:  # pragma: no cover - runtime preflight only
        print(f"Torchvision import failed: {exc}")
        return 1

    print(f"Torchvision: {torchvision.__version__}")

    try:
        import vllm
    except Exception as exc:  # pragma: no cover - runtime preflight only
        print(f"vLLM import failed: {exc}")
        return 1

    print(f"vLLM: {vllm.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
