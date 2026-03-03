import os
from dotenv import load_dotenv

load_dotenv()

UPLOAD_DIR = "sessions"
VIDEO_MODEL_PATH = "models/video_fake_detector.h5"
AUDIO_MODEL_PATH = "models/audio_fake_detector.pth"
DATABASE_URL = os.getenv("DATABASE_URL")
VIDEO_PATH = "D:/aixoware/fAIk-backend/helpers/input.mp4"
DEVICE = "cpu"
CHECKPOINT = "checkpoints/fused_best.pt"  
THRESHOLD = 0.5    
OTP_EXPIRE_MINUTES = 5
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")