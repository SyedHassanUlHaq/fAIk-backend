import cv2
import numpy as np
import logging
from collections import deque
from skimage.metrics import structural_similarity as ssim

from .embeddings import compute_embeddings_batch, cosine_similarity

logger = logging.getLogger(__name__)

def mse(a, b):
    try:
        return np.mean((a.astype("float") - b.astype("float")) ** 2)
    except Exception as e:
        logger.error(f"Error calculating MSE: {e}", exc_info=True)
        return 0.0

def detect_scene_changes(video_path, model, processor, device, compute_emb_only_on_candidates=True):
    try:
        cap = cv2.VideoCapture(video_path)

        ret, prev_frame = cap.read()
        if not ret:
            logger.error(f"Could not read first frame from {video_path}")
            raise ValueError("Could not read first frame")

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        reference_frames = [prev_frame]  
        candidates = []                  
        frame_idx = 1

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ssim_score, _ = ssim(prev_gray, gray, full=True)
                mse_score = mse(prev_gray, gray)

                is_candidate = not (compute_emb_only_on_candidates and (ssim_score >= 0.7 or mse_score <= 3000))

                if is_candidate:
                    candidates.append((frame_idx, ssim_score, mse_score))
                    reference_frames.append(frame)

                prev_gray = gray
                frame_idx += 1
            except Exception as e:
                logger.warning(f"Error processing frame {frame_idx}: {e}")
                continue

        cap.release()

        if not candidates:
            logger.info(f"No scene changes detected in {video_path}")
            return []

        try:
            embeddings = compute_embeddings_batch(reference_frames, model, processor, device)
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}", exc_info=True)
            return []

        results = []
        window = deque(maxlen=20)

        for i, (cand_frame_idx, ssim_score, mse_score) in enumerate(candidates):
            try:
                prev_emb = embeddings[i]
                curr_emb = embeddings[i + 1]

                emb_diff = 1 - cosine_similarity(prev_emb, curr_emb)
                window.append(emb_diff)

                adaptive_thresh = min(0.15, np.mean(window) + 2 * np.std(window)) if len(window) > 5 else 0.15

                metrics_triggered = sum([
                    ssim_score < 0.7,
                    mse_score > 3000,
                    emb_diff > adaptive_thresh,
                ])
                if metrics_triggered >= 2:
                    results.append({
                        "frame": cand_frame_idx,
                        "ssim": float(ssim_score),
                        "mse": float(mse_score),
                        "emb_diff": float(emb_diff),
                    })
            except Exception as e:
                logger.warning(f"Error processing candidate at frame {cand_frame_idx}: {e}")
                continue

        return results
    except Exception as e:
        logger.error(f"Error detecting scene changes in {video_path}: {e}", exc_info=True)
        raise