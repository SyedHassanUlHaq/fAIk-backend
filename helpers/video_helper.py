import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed


def create_chunk(input_path, start, end, chunk_path):
    """Create a single video chunk using ffmpeg"""
    print(f"[INFO] Creating chunk: {chunk_path} (from {start}s to {end}s)")
    cmd = [
        "ffmpeg",
        "-y",  # overwrite output if exists
        "-i", input_path,
        "-ss", str(start),
        "-to", str(end),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-strict", "experimental",
        "-loglevel", "error",
        chunk_path
    ]
    subprocess.run(cmd, check=True)
    print(f"[INFO] Chunk created: {chunk_path}")
    return chunk_path


def split_video_into_chunks(input_path, output_dir, chunk_length=5):
    """
    Split video into chunks of chunk_length seconds using ffmpeg, preserving audio.
    Returns list of chunk paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[INFO] Getting video duration for: {input_path}")
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", input_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    duration = float(result.stdout)
    print(f"[INFO] Video duration: {duration:.2f} seconds")

    # Prepare chunk info
    chunk_infos = []
    for start in range(0, int(duration), chunk_length):
        end = min(start + chunk_length, duration)
        chunk_path = os.path.join(output_dir, f"chunk_{start}-{end}.mp4")
        chunk_infos.append((input_path, start, end, chunk_path))

    chunks = []

    # Run chunk creation in parallel
    max_workers = min(len(chunk_infos), os.cpu_count())
    print(f"[INFO] Starting parallel chunk creation with {max_workers} workers")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(create_chunk, *info) for info in chunk_infos]
        for future in as_completed(futures):
            chunks.append(future.result())

    chunks.sort()  # optional: ensure chunks are in chronological order
    print(f"[INFO] Total chunks created: {len(chunks)}")
    return chunks