#!/usr/bin/env python3
"""
Benchmark inference time and memory usage for the video deepfake
and scene detection models.

Usage:
    .venv/bin/python benchmark.py /path/to/video.mp4
"""

import os
import sys
import time
import tempfile

import psutil
import torch


# ----------------------------------------------------------------
# Memory helpers
# ----------------------------------------------------------------

_process = psutil.Process()


def _ram_mb() -> float:
    return _process.memory_info().rss / 1024 ** 2


def _vram_mb() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        torch.cuda.memory_allocated() / 1024 ** 2,
        torch.cuda.memory_reserved() / 1024 ** 2,
    )


def _mem_snapshot(label: str):
    ram = _ram_mb()
    alloc, reserved = _vram_mb()
    if torch.cuda.is_available():
        print(f"  [{label}] RAM: {ram:.0f} MB | VRAM alloc: {alloc:.0f} MB | VRAM reserved: {reserved:.0f} MB")
    else:
        print(f"  [{label}] RAM: {ram:.0f} MB | (no GPU)")


# ----------------------------------------------------------------
# Main benchmark
# ----------------------------------------------------------------

def run(video_path: str):
    print(f"\nVideo : {video_path}")
    print(f"Device: {'cuda (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 65)

    baseline_ram = _ram_mb()
    baseline_vram, _ = _vram_mb()

    # ----------------------------------------------------------------
    # Video deepfake model
    # ----------------------------------------------------------------
    print("\n--- Video deepfake model (RAFT + XCLIP + DeMamba + FusedHead) ---")
    _mem_snapshot("before load")

    t0 = time.perf_counter()
    from ml_models.video import load_models
    load_models()
    load_time = time.perf_counter() - t0

    _mem_snapshot("after load ")
    vram_after_video, _ = _vram_mb()
    ram_after_video = _ram_mb()
    print(f"  Load time : {load_time:.1f}s")
    print(f"  RAM delta : +{ram_after_video - baseline_ram:.0f} MB")
    if torch.cuda.is_available():
        print(f"  VRAM delta: +{vram_after_video - baseline_vram:.0f} MB")

    # Inference
    from helpers.video_helper import split_video_into_chunks, infer_chunk
    chunks_dir = tempfile.mkdtemp()
    chunks = split_video_into_chunks(video_path, chunks_dir, chunk_length=5)
    print(f"\n  Chunks: {len(chunks)} × 5s")

    _mem_snapshot("before infer")
    t0 = time.perf_counter()
    chunk_results = [infer_chunk(c) for c in chunks]
    video_infer_time = time.perf_counter() - t0
    _mem_snapshot("after infer ")

    probs = [r["result"].get("probability", 0.0) for r in chunk_results]
    print(f"  Per-chunk probabilities : {[round(p, 3) for p in probs]}")
    print(f"  Overall fake probability: {sum(probs)/len(probs):.3f}")
    print(f"  Inference time: {video_infer_time:.2f}s  ({video_infer_time/len(chunks):.2f}s per chunk)")

    for c in chunks:
        try:
            os.remove(c)
        except OSError:
            pass
    try:
        os.rmdir(chunks_dir)
    except OSError:
        pass

    # ----------------------------------------------------------------
    # Scene detection model
    # ----------------------------------------------------------------
    print("\n--- Scene detection model (nomic-embed-vision-v1.5) ---")
    _mem_snapshot("before load")

    t0 = time.perf_counter()
    from ml_models.scene_detection import get_embedding_model
    scene_model, scene_processor, scene_device = get_embedding_model()
    load_time = time.perf_counter() - t0

    _mem_snapshot("after load ")
    vram_after_scene, _ = _vram_mb()
    ram_after_scene = _ram_mb()
    print(f"  Load time : {load_time:.1f}s")
    print(f"  RAM delta : +{ram_after_scene - ram_after_video:.0f} MB")
    if torch.cuda.is_available():
        print(f"  VRAM delta: +{vram_after_scene - vram_after_video:.0f} MB")

    from services.scene_detection.video_utils import convert_to_fps
    from services.scene_detection.detector import detect_scene_changes
    converted = convert_to_fps(video_path)

    _mem_snapshot("before infer")
    t0 = time.perf_counter()
    cuts = detect_scene_changes(converted, scene_model, scene_processor, scene_device)
    scene_infer_time = time.perf_counter() - t0
    _mem_snapshot("after infer ")

    print(f"  Scene cuts detected: {len(cuts)}")
    print(f"  Inference time: {scene_infer_time:.2f}s")

    if converted != video_path and os.path.exists(converted):
        os.remove(converted)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print(f"\n{'=' * 65}")
    print(f"  {'Model':<30} {'Inference':>10}  {'RAM':>10}  {'VRAM':>10}")
    print(f"  {'-'*62}")
    print(f"  {'Video deepfake':<30} {video_infer_time:>9.2f}s  {ram_after_video - baseline_ram:>8.0f}MB  {vram_after_video - baseline_vram:>8.0f}MB")
    print(f"  {'Scene detection':<30} {scene_infer_time:>9.2f}s  {ram_after_scene - ram_after_video:>8.0f}MB  {vram_after_scene - vram_after_video:>8.0f}MB")
    print(f"  {'-'*62}")
    total_ram = ram_after_scene - baseline_ram
    total_vram = vram_after_scene - baseline_vram
    total_time = video_infer_time + scene_infer_time
    print(f"  {'TOTAL (both loaded)':<30} {total_time:>9.2f}s  {total_ram:>8.0f}MB  {total_vram:>8.0f}MB")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: .venv/bin/python benchmark.py /path/to/video.mp4")
        sys.exit(1)
    run(sys.argv[1])
