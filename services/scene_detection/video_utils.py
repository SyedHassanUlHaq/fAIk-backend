import cv2
import os
import hashlib

FPS20_DIR = "20 FPS Videos"

def convert_to_fps(video_path, target_fps=20):
    """Convert video to target FPS (default 20 fps)."""
    os.makedirs(FPS20_DIR, exist_ok=True)

    abs_path = os.path.abspath(video_path)
    path_hash = hashlib.md5(abs_path.encode()).hexdigest()[:8]
    base_name = os.path.basename(video_path)
    name, ext = os.path.splitext(base_name)
    new_path = os.path.join(FPS20_DIR, f"{name}_{path_hash}_{target_fps}fps{ext}")

    if os.path.exists(new_path):
        return new_path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video")

    width = int(cap.get(3))
    height = int(cap.get(4))

    out = cv2.VideoWriter(
        new_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        target_fps,
        (width, height)
    )

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if original_fps <= target_fps:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
    else:
        interval = int(round(original_fps / target_fps))
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i % interval == 0:
                out.write(frame)
            i += 1

    cap.release()
    out.release()

    return new_path