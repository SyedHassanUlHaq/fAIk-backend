from transformers import pipeline

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = pipeline(
            "image-feature-extraction",
            model="nomic-ai/nomic-embed-vision-v1.5",
            trust_remote_code=True,
            device=-1  # CPU
        )
    return _embedding_model