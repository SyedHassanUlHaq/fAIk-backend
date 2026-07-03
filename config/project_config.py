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

# --- JWT ---
SECRET_KEY = os.getenv("SECRET_KEY")
REFRESH_SECRET_KEY = os.getenv("REFRESH_SECRET_KEY", os.getenv("SECRET_KEY"))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 30

# --- Email / OTP ---
OTP_EXPIRE_MINUTES = 5
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

# --- OAuth ---
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
APPLE_CLIENT_ID = os.getenv("APPLE_CLIENT_ID")   # Bundle ID / Service ID

# --- AWS S3 ---
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "faik-storage")
CDN_BASE_URL = os.getenv("CDN_BASE_URL", "")  # optional CloudFront domain

# --- Redis / Celery ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# --- Stripe ---
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

# --- Plan limits ---
PLAN_SCAN_LIMITS: dict[str, int | None] = {
    "free": 50,
    "pro": 500,
    "team": None,   # unlimited
}
