from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuth
from sqlalchemy.orm import Session
from ..database import get_db
from .. import schemas, crud
from ..utils.jwt import create_access_token
from ..utils.email import send_otp_email, generate_otp
from ..utils.otp_store import store_otp, verify_otp, delete_otp
from ..utils.security import hash_password
from ..schemas import OTPVerify
import asyncio
import random, string
import os
from dotenv import load_dotenv

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/signup-request")
async def signup_request(user: schemas.OTPRequest, db: Session = Depends(get_db)):
    if crud.get_user_by_email(db, user.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    if user.password != user.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    otp = generate_otp()
    store_otp(user.email, otp, user.dict(exclude={"confirm_password"}))
    await send_otp_email(user.email, otp)
    return {"detail": "OTP sent to your email. It will expire in 5 minutes."}

@router.post("/verify-otp")
def verify_otp_endpoint(payload: schemas.OTPVerify, db: Session = Depends(get_db)):
    valid, data_or_msg = verify_otp(payload.email, payload.otp)
    if not valid:
        raise HTTPException(status_code=400, detail=data_or_msg)

    user_data = data_or_msg
    db_user = crud.create_user(db, schemas.UserCreate(**user_data, confirm_password=user_data["password"]))

    delete_otp(payload.email)

    return {"detail": "User created successfully", "user": {
        "id": db_user.id,
        "first_name": db_user.first_name,
        "last_name": db_user.last_name,
        "email": db_user.email
    }}
    
@router.post("/login", response_model=schemas.Token)
def login(user: schemas.UserLogin, db: Session = Depends(get_db)):
    authenticated_user = crud.authenticate_user(db, user.email, user.password)

    if not authenticated_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    access_token = create_access_token(
        data={"sub": str(authenticated_user.id)}
    )

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }
    
@router.post("/reset-password-request")
async def reset_password_request(payload: schemas.ResetPasswordRequest, db: Session = Depends(get_db)):
    user = crud.get_user_by_email(db, payload.email)
    if not user:
        raise HTTPException(status_code=400, detail="Email not registered")

    otp = generate_otp()
    store_otp(payload.email, otp, {"email": payload.email})

    await send_otp_email(payload.email, otp)
    return {"detail": "OTP sent to your email. It will expire in 5 minutes."}


@router.post("/reset-password-verify")
def reset_password_verify(payload: schemas.ResetPasswordVerify, db: Session = Depends(get_db)):
    valid, data_or_msg = verify_otp(payload.email, payload.otp)
    if not valid:
        raise HTTPException(status_code=400, detail=data_or_msg)

    user = crud.get_user_by_email(db, payload.email)
    if not user:
        raise HTTPException(status_code=400, detail="User not found")

    user.hashed_password = hash_password(payload.new_password)
    db.commit()

    delete_otp(payload.email)

    return {"detail": "Password reset successfully"}

load_dotenv()

oauth = OAuth()
CONF_URL = "https://accounts.google.com/.well-known/openid-configuration"

# Configure OAuth

# Google
oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url=CONF_URL,
    client_kwargs={"scope": "openid email profile"},
)

# Facebook
oauth.register(
    name="facebook",
    client_id=os.getenv("FACEBOOK_CLIENT_ID"),
    client_secret=os.getenv("FACEBOOK_CLIENT_SECRET"),
    access_token_url="https://graph.facebook.com/v16.0/oauth/access_token",
    authorize_url="https://www.facebook.com/v16.0/dialog/oauth",
    client_kwargs={"scope": "email public_profile"},
)

# Microsoft / Outlook
oauth.register(
    name="microsoft",
    client_id=os.getenv("MICROSOFT_CLIENT_ID"),
    client_secret=os.getenv("MICROSOFT_CLIENT_SECRET"),
    access_token_url="https://login.microsoftonline.com/common/oauth2/v2.0/token",
    authorize_url="https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
    client_kwargs={"scope": "User.Read"},
)

router = APIRouter(prefix="/auth", tags=["Authentication"])

OAUTH_PROVIDERS = ["google", "facebook", "microsoft"]

@router.get("/{provider}/login")
async def oauth_login(provider: str, request: Request):
    if provider not in OAUTH_PROVIDERS:
        raise HTTPException(status_code=400, detail="Unsupported provider")
    redirect_uri = os.getenv(f"{provider.upper()}_REDIRECT_URI")
    return await oauth.create_client(provider).authorize_redirect(request, redirect_uri)


@router.get("/{provider}/callback")
async def oauth_callback(provider: str, request: Request, db: Session = Depends(get_db)):
    if provider not in OAUTH_PROVIDERS:
        raise HTTPException(status_code=400, detail="Unsupported provider")

    token = await oauth.create_client(provider).authorize_access_token(request)
    
    if provider == "google":
        user_info = token.get("userinfo") or await oauth.google.parse_id_token(request, token)
        email = user_info["email"]
        first_name = user_info.get("given_name", "")
        last_name = user_info.get("family_name", "")
    elif provider == "facebook":
        resp = await oauth.facebook.get("https://graph.facebook.com/me?fields=id,email,first_name,last_name", token=token)
        data = resp.json()
        email = data["email"]
        first_name = data["first_name"]
        last_name = data["last_name"]
    elif provider == "microsoft":
        resp = await oauth.microsoft.get("https://graph.microsoft.com/v1.0/me", token=token)
        data = resp.json()
        email = data.get("userPrincipalName", f"user_{data['id']}@example.com")
        first_name = data.get("givenName", "")
        last_name = data.get("surname", "")

    user = crud.get_user_by_email(db, email)
    if not user:
        random_pass = "".join(random.choices(string.ascii_letters + string.digits, k=12))
        user = crud.create_user(db, schemas.UserCreate(
            first_name=first_name,
            last_name=last_name,
            email=email,
            password=random_pass,
            confirm_password=random_pass
        ))

    access_token = create_access_token(data={"sub": str(user.id)})

    return {"access_token": access_token, "token_type": "bearer", "user": {
        "id": user.id,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "email": user.email
    }}
