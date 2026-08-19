from __future__ import annotations

import platform
from importlib import metadata

import torch


def main() -> None:
    print(f"architecture={platform.machine()}")
    print(f"python={platform.python_version()}")
    for package in ("torch", "transformers", "qwen-vl-utils", "accelerate", "triton", "av"):
        print(f"{package}={metadata.version(package)}")
    print(f"torch.version.cuda={torch.version.cuda}")
    print(f"cuda.available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda.device={torch.cuda.get_device_name(0)}")
        print(f"cuda.capability={torch.cuda.get_device_capability(0)}")

    if platform.python_version() != "3.12.3":
        raise SystemExit(f"Expected pinned Python 3.12.3, got {platform.python_version()}")
    if platform.machine() not in {"aarch64", "arm64"}:
        raise SystemExit("This lock/setup is intended for the ARM64 GB10 baseline")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if torch.cuda.get_device_capability(0) != (12, 1):
        raise SystemExit("Expected GB10 compute capability 12.1")


if __name__ == "__main__":
    main()
