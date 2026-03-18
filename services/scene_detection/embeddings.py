import numpy as np
import cv2
from PIL import Image
from numpy.linalg import norm

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def compute_embedding(frame, pipe):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    emb = pipe(pil_image)[0]
    return np.mean(np.array(emb), axis=0)