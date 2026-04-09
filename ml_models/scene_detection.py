import threading
from transformers import AutoModel, AutoProcessor
import torch

_load_lock = threading.Lock()

def get_embedding_model():
    global _embedding_model, _embedding_processor, _embedding_device

    with _load_lock:
        if _embedding_model is None:
            print("[*] Loading embedding model...")

            model_name = "nomic-ai/nomic-embed-vision-v1.5"

            _embedding_device = "cuda" if torch.cuda.is_available() else "cpu"

            _embedding_processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            _embedding_model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True
            ).to(_embedding_device)

            _embedding_model.eval()

            print(f"[+] Embedding model loaded on {_embedding_device}")

    return _embedding_model, _embedding_processor, _embedding_device
