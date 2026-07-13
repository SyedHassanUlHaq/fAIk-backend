import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import video
from ml_models.video import load_models
from utils.errors import AppError

load_dotenv()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print("[*] Loading video AI models at startup...")
    load_models()
    print("[+] Video AI models loaded")
    yield
    print("[*] Video AI service shutdown")


app = FastAPI(title="5dot Video AI Service", version="1.0", lifespan=lifespan)

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


app.include_router(video.router, prefix="/v1/video", tags=["Video AI"])


@app.get("/")
def root():
    return {"message": "5dot Video AI Service running", "version": app.version}
