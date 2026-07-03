import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware

from api.v1 import auth, feedback, plans, scans, stats, subscriptions, users, webhooks
# from ml_models.video import load_models
# from ml_models.scene_detection import get_embedding_model
from utils.errors import AppError

load_dotenv()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # print("[*] Loading models at startup...")
    # load_models()
    # model, processor, device = get_embedding_model()
    # app.state.embedding_model = model
    # app.state.embedding_processor = processor
    # app.state.embedding_device = device
    # print("[+] All models loaded")
    yield
    print("[*] Server shutdown")


app = FastAPI(title="fAIk API", version="1.0", lifespan=lifespan)

app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET_KEY", "super-secret-key"))
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


# Routes
app.include_router(auth.router,          prefix="/v1/auth",          tags=["Auth"])
app.include_router(users.router,         prefix="/v1/users",         tags=["Users"])
app.include_router(scans.router,         prefix="/v1/scans",         tags=["Scans"])
app.include_router(feedback.router,      prefix="/v1/feedback",      tags=["Feedback"])
app.include_router(plans.router,         prefix="/v1/plans",         tags=["Plans"])
app.include_router(subscriptions.router, prefix="/v1/subscriptions", tags=["Subscriptions"])
app.include_router(stats.router,         prefix="/v1/stats",         tags=["Stats"])
app.include_router(webhooks.router,      prefix="/v1/webhooks",      tags=["Webhooks"])


@app.get("/")
def root():
    return {"message": "fAIk API running", "version": app.version}
