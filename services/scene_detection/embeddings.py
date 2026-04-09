import torch
import logging
import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

def cosine_similarity(a, b):
    try:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    except Exception as e:
        logger.error(f"Error computing cosine similarity: {e}", exc_info=True)
        return 0.0

def compute_embedding(frame, model, processor, device="cpu"):
    try:
        if frame is None:
            raise ValueError("Frame is None")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        inputs = processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        emb = outputs.last_hidden_state.mean(dim=1)
        return emb.squeeze().cpu().numpy()
    except Exception as e:
        logger.error(f"Error computing embedding: {e}", exc_info=True)
        raise

def compute_embeddings_batch(frames, model, processor, device="cpu"):
    """Compute embeddings for multiple frames in a single forward pass."""
    try:
        pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        inputs = processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)  # (B, D)
        return emb.cpu().numpy()  # (B, D)
    except Exception as e:
        logger.error(f"Error computing batch embeddings: {e}", exc_info=True)
        raise