from datetime import datetime, timedelta
import logging
from jose import jwt, JWTError
import os
from dotenv import load_dotenv
from config.project_config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES

load_dotenv()

logger = logging.getLogger(__name__)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    try:
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    except JWTError as e:
        logger.error(f"Failed to encode JWT token: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating access token: {e}", exc_info=True)
        raise