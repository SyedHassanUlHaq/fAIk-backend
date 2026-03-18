import cv2
import numpy as np
from collections import deque
from skimage.metrics import structural_similarity as ssim

from .embeddings import compute_embedding, cosine_similarity

def mse(a, b):
    return np.mean((a.astype("float") - b.astype("float")) ** 2)

def detect_scene_changes(video_path, pipe):
    cap = cv2.VideoCapture(video_path)

    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_emb = compute_embedding(prev_frame, pipe)

    results = []
    window = deque(maxlen=20)
    frame_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ssim_score, _ = ssim(prev_gray, gray, full=True)
        mse_score = mse(prev_gray, gray)

        emb = compute_embedding(frame, pipe)
        emb_diff = 1 - cosine_similarity(prev_emb, emb)

        window.append(emb_diff)

        adaptive_thresh = max(0.15, np.mean(window) + 2*np.std(window)) if len(window) > 5 else 0.15

        if ssim_score < 0.7 and mse_score > 3000 and emb_diff > adaptive_thresh:
            results.append({
                "frame": frame_idx,
                "ssim": float(ssim_score),
                "mse": float(mse_score),
                "emb_diff": float(emb_diff)
            })

        prev_gray = gray
        prev_emb = emb
        frame_idx += 1

    cap.release()
    return results