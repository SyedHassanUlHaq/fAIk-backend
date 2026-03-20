import torch
import numpy as np
import cv2
from PIL import Image
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
def compute_embedding(frame, model, processor, device="cpu"):
    if frame is None:
        raise ValueError("Frame is None")

    # Convert BGR → RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # 🔥 THIS replaces pipeline preprocessing
    inputs = processor(images=pil_image, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # 👇 Most vision embedding models use last_hidden_state
    emb = outputs.last_hidden_state

    # Pool to single vector
    emb = emb.mean(dim=1)

    return emb.squeeze().cpu().numpy()