from transformers import pipeline

def get_embedding_model():
    pipe = pipeline(
        "image-feature-extraction",
        model="nomic-ai/nomic-embed-vision-v1.5",
        trust_remote_code=True,
        device=-1  # CPU
    )
    return pipe