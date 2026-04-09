from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware
import os
import logging
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from api.v1 import payments, webhooks, auth, video, scene
from ml_models.video import load_models
from ml_models.scene_detection import get_embedding_model
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

logger = logging.getLogger(__name__)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("[*] Loading models at startup...")
        load_models()
        try:
            model, processor, device = get_embedding_model()
            app.state.embedding_model = model
            app.state.embedding_processor = processor
            app.state.embedding_device = device
            print("[+] All models loaded")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            app.state.embedding_model = None
            app.state.embedding_processor = None
            app.state.embedding_device = None
    except Exception as e:
        logger.error(f"Failed to load video models: {e}", exc_info=True)
    
    yield
    
    try:
        print("[*] Server shutdown")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)

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

origins = [
    "https://syntheticvideodetector.netlify.app",
]