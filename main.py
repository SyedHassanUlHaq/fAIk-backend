from fastapi import FastAPI
from .database import engine
from . import models
from .api.v1 import payments, webhooks, auth, video
from starlette.middleware.sessions import SessionMiddleware
import os
from dotenv import load_dotenv

load_dotenv()

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Auth API")

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY", "super-secret-key")
)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(payments.router, prefix="/api/v1/payments", tags=["Payments"])
app.include_router(webhooks.router, prefix="/api/v1/webhooks", tags=["Webhooks"])
app.include_router(video.router, prefix="/api/v1/video", tags=["Video Processing"])

@app.get("/")
def root():
    return {"message": "API running"}