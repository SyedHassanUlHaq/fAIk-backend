from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from api.v1 import payments, webhooks, auth, video, scene
from ml_models.video import load_models

load_dotenv()

# Set PyTorch CUDA memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[*] Loading models at startup...")
    load_models()
    yield
    print("[*] Server shutdown")

app = FastAPI(title="fAIk Backend API", version="1.0", lifespan=lifespan)

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY", "super-secret-key")
)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(payments.router, prefix="/api/v1/payments", tags=["Payments"])
app.include_router(webhooks.router, prefix="/api/v1/webhooks", tags=["Webhooks"])
app.include_router(video.router, prefix="/api/v1/video", tags=["Video Processing"])
app.include_router(scene.router, prefix="/api/v1/scene", tags=["Scene Detection"])

@app.get("/")
def root():
    return {"message": "API running", "version": app.version}