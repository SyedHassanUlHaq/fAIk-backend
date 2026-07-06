import os
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
UPLOAD_DIR = "sessions"
CHECKPOINT = "checkpoints/fused_best.pt"

# --- ML ---
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"
THRESHOLD = 0.5
MODEL_VERSION = os.getenv("MODEL_VERSION", "v3.2")

# --- Database ---
DATABASE_URL = os.getenv("DATABASE_URL")

# --- AWS S3 ---
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "faik-storage")
CDN_BASE_URL = os.getenv("CDN_BASE_URL", "")  # optional CloudFront domain

# --- Redis / Celery ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
