from fastapi import FastAPI
from .database import engine
from . import models
from .routers import auth
from .routers import payments, webhooks
from .config import stripe
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

app.include_router(auth.router)
app.include_router(payments.router)
app.include_router(webhooks.router)

@app.get("/")
def root():
    return {"message": "API running"}