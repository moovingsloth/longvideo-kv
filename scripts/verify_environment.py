from __future__ import annotations

import platform
import os
import subprocess
from importlib import metadata

import torch
import torchvision


def main() -> None:
    print(f"architecture={platform.machine()}")
    print(f"python={platform.python_version()}")
    for package in ("torch", "torchvision", "torchcodec", "transformers", "qwen-vl-utils", "accelerate", "triton", "av"):
        print(f"{package}={metadata.version(package)}")
    ffmpeg = subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    print(ffmpeg)
    print(f"torchvision.io.read_video={hasattr(torchvision.io, 'read_video')}")
    print(f"torch.version.cuda={torch.version.cuda}")
    print(f"cuda.available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda.device={torch.cuda.get_device_name(0)}")
        print(f"cuda.capability={torch.cuda.get_device_capability(0)}")

    if platform.python_version() != "3.12.3":
        raise SystemExit(f"Expected pinned Python 3.12.3, got {platform.python_version()}")

    arch = platform.machine()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    capability = torch.cuda.get_device_capability(0)

    if arch in {"aarch64", "arm64"}:
        if capability != (12, 1):
            raise SystemExit(f"Expected GB10 compute capability 12.1, got {capability}")
        print("✓ Verified GB10 environment (ARM64, H100)")
    elif arch == "x86_64":
        if capability != (8, 6):
            raise SystemExit(f"Expected RTX 3090 compute capability 8.6, got {capability}")
        print("✓ Verified RTX 3090 environment (x86_64)")
    else:
        raise SystemExit(f"Unsupported architecture: {arch}. Expected aarch64/arm64 (GB10) or x86_64 (RTX 3090)")

    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")
    import qwen_vl_utils.vision_process as qwen_vision_process

    qwen_vision_process.get_video_reader_backend.cache_clear()
    video_reader = qwen_vision_process.get_video_reader_backend()
    print(f"qwen_vl_utils.video_reader={video_reader}")
    if video_reader != "torchcodec":
        raise SystemExit(f"Expected qwen-vl-utils video reader torchcodec, got {video_reader}")


if __name__ == "__main__":
    main()
