import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import scene
from ml_models.scene_detection import get_embedding_model
from utils.errors import AppError

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[*] Loading scene/tamper detection AI model at startup...")
    model, processor, device = get_embedding_model()
    app.state.embedding_model = model
    app.state.embedding_processor = processor
    app.state.embedding_device = device
    print("[+] Scene detection model loaded")
    yield
    print("[*] Scene detection service shutdown")


app = FastAPI(title="fAIk Scene Detection Service", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AppError)
async def app_error_handler(_request: Request, exc: AppError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message, "statusCode": exc.status_code}},
    )


app.include_router(scene.router, prefix="/v1/scene", tags=["Scene / Tamper AI"])


@app.get("/")
def root():
    return {"message": "fAIk Scene Detection Service running", "version": app.version}
