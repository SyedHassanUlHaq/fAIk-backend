import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import voice
from utils.errors import AppError

load_dotenv()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    print("[*] Voice AI service shutdown")


app = FastAPI(title="5dot Voice AI Service", version="1.0", lifespan=lifespan)

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


app.include_router(voice.router, prefix="/v1/voice", tags=["Voice AI"])


@app.get("/")
def root():
    return {"message": "5dot Voice AI Service running", "version": app.version}
