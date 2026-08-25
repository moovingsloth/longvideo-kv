#!/usr/bin/env python3
"""Preflight LongVideoBench video decode with the evaluation video backend."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--per-doc", action="store_true")
    parser.add_argument("--output-path", default="results/video-decode-preflight.jsonl")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def configure_video_reader() -> str:
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")
    os.environ.setdefault("STRICT_VIDEO_READER", "1")
    os.environ.setdefault("TORCHCODEC_NUM_THREADS", "8")

    import qwen_vl_utils.vision_process as qwen_vision_process

    qwen_vision_process.get_video_reader_backend.cache_clear()
    selected_reader = qwen_vision_process.get_video_reader_backend()
    forced_reader = os.environ.get("FORCE_QWENVL_VIDEO_READER")
    if forced_reader and selected_reader != forced_reader:
        raise RuntimeError(f"Expected qwen-vl-utils video reader {forced_reader!r}, got {selected_reader!r}")

    if selected_reader not in qwen_vision_process.VIDEO_READER_BACKENDS:
        raise RuntimeError(f"Unknown qwen-vl-utils video reader backend: {selected_reader!r}")

    strict_reader = os.environ.get("STRICT_VIDEO_READER", "").lower() in {"1", "true", "yes"}
    if strict_reader and selected_reader != "torchvision":
        def _fail_torchvision_fallback(ele: dict[str, Any]) -> None:
            video = ele.get("video", "<unknown>")
            raise RuntimeError(
                f"Strict video reader is enabled: {selected_reader!r} failed for {video!r}; "
                "refusing qwen-vl-utils torchvision fallback."
            )

        qwen_vision_process.VIDEO_READER_BACKENDS["torchvision"] = _fail_torchvision_fallback

    return selected_reader


def jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return str(value)


def safe_nframes(video_path: Path, target_frames: int, reader: str) -> tuple[int, int | None]:
    total_frames: int | None = None
    if reader == "torchcodec":
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(str(video_path), num_ffmpeg_threads=1)
        total_frames = int(decoder.metadata.num_frames)
    elif reader == "decord":
        import decord

        total_frames = len(decord.VideoReader(str(video_path)))

    if total_frames is None:
        return target_frames, None

    nframes = min(int(target_frames), total_frames)
    nframes = (nframes // 2) * 2
    return max(2, nframes), total_frames


def make_video_content(args: argparse.Namespace, video_path: Path, reader: str) -> tuple[dict[str, Any], int | None]:
    content: dict[str, Any] = {
        "type": "video",
        "video": str(video_path),
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
    }
    if args.fps is None:
        nframes, total_frames = safe_nframes(video_path, args.frames, reader)
        content["nframes"] = nframes
        return content, total_frames

    content["fps"] = args.fps
    content["max_frames"] = args.frames
    return content, None


def frame_index_summary(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    frame_indices = jsonable(metadata.get("frames_indices", []))
    if not isinstance(frame_indices, list):
        frame_indices = []
    return {
        "backend": metadata.get("video_backend"),
        "fps": jsonable(metadata.get("fps")),
        "total_num_frames": jsonable(metadata.get("total_num_frames")),
        "sampled_frames": len(frame_indices),
        "frame_indices_head": frame_indices[:3],
        "frame_indices_tail": frame_indices[-3:] if len(frame_indices) >= 3 else frame_indices,
    }


def check_one(args: argparse.Namespace, processor: Any, reader: str, video_path: Path) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    st = time.time()
    content, probed_total_frames = make_video_content(args, video_path, reader)
    messages = [
        {
            "role": "user",
            "content": [
                content,
                {"type": "text", "text": "Answer with the option's letter from the given choices directly."},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, _ = process_vision_info(
        messages,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    video_metadatas = None
    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadatas,
        padding=True,
        return_tensors="pt",
    )

    grid = None
    visual_tokens = None
    if "video_grid_thw" in inputs:
        grid = [int(value) for value in inputs["video_grid_thw"][0].tolist()]
        merge_size = getattr(processor.image_processor, "merge_size", 2)
        visual_tokens = grid[0] * grid[1] * grid[2] // (merge_size**2)

    return {
        "ok": True,
        "elapsed_sec": round(time.time() - st, 3),
        "video_path": str(video_path),
        "requested_frames": args.frames,
        "requested_fps": args.fps,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "reader": reader,
        "probed_total_frames": probed_total_frames,
        "video_grid_thw": grid,
        "visual_tokens": visual_tokens,
        "metadata": frame_index_summary(video_metadatas[0] if video_metadatas else None),
    }


def load_longvideobench() -> tuple[Any, Path]:
    from datasets import DownloadConfig, load_dataset

    from lmms_eval.tasks.longvideobench.utils import _load_task_config, _resolve_dataset_dir

    task_config = _load_task_config("longvideobench_val_v.yaml")
    dataset_kwargs = task_config["dataset_kwargs"].copy()
    load_kwargs = {
        key: dataset_kwargs[key]
        for key in ("cache_dir", "revision", "token")
        if key in dataset_kwargs
    }
    if dataset_kwargs.get("local_files_only"):
        load_kwargs["download_config"] = DownloadConfig(local_files_only=True)
    if dataset_kwargs.get("force_download"):
        load_kwargs["download_mode"] = "force_redownload"

    dataset = load_dataset(
        task_config["dataset_path"],
        split=task_config["test_split"],
        **load_kwargs,
    )
    video_dir, _ = _resolve_dataset_dir("longvideobench_val_v.yaml", "video_subdir", "videos/")
    return dataset, Path(video_dir)


def iter_video_docs(dataset: Any, video_dir: Path, limit: int | None, per_doc: bool):
    seen: set[str] = set()
    emitted = 0
    for doc_index, doc in enumerate(dataset):
        rel_path = doc["video_path"]
        if not per_doc:
            if rel_path in seen:
                continue
            seen.add(rel_path)
        yield doc_index, doc, video_dir / rel_path
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def main() -> int:
    args = parse_args()
    if args.frames < 2:
        raise ValueError("--frames must be >= 2")
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")

    reader = configure_video_reader()

    from transformers import AutoProcessor

    dataset, video_dir = load_longvideobench()
    processor = AutoProcessor.from_pretrained(
        args.pretrained,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    failures = 0
    checked = 0
    planned_docs = list(iter_video_docs(dataset, video_dir, args.limit, args.per_doc))
    print(
        "decode preflight: "
        f"reader={reader} videos={len(planned_docs)} repeat={args.repeat} "
        f"frames={args.frames} fps={args.fps} min_pixels={args.min_pixels} max_pixels={args.max_pixels}",
        flush=True,
    )

    with output_path.open("w") as f:
        for repeat_index in range(args.repeat):
            for sequence_index, (doc_index, doc, video_path) in enumerate(planned_docs):
                base_record = {
                    "repeat_index": repeat_index,
                    "sequence_index": sequence_index,
                    "doc_index": doc_index,
                    "id": doc.get("id"),
                    "video_id": doc.get("video_id"),
                    "relative_video_path": doc.get("video_path"),
                    "video_path": str(video_path),
                }
                try:
                    if not video_path.exists():
                        raise FileNotFoundError(str(video_path))
                    record = {**base_record, **check_one(args, processor, reader, video_path)}
                except Exception as exc:
                    failures += 1
                    record = {
                        **base_record,
                        "ok": False,
                        "reader": reader,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    print(
                        f"FAIL repeat={repeat_index} index={sequence_index} doc={doc.get('id')} video={video_path}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                checked += 1
                f.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")
                f.flush()

                if args.progress_every > 0 and checked % args.progress_every == 0:
                    print(f"checked={checked} failures={failures}", flush=True)

    print(f"decode preflight done: checked={checked} failures={failures} output={output_path}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
