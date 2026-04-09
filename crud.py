from sqlalchemy.orm import Session
import logging
import models
import schemas.schemas as schemas
from utils.security import hash_password, verify_password

logger = logging.getLogger(__name__)

def get_user_by_email(db: Session, email: str):
    try:
        return db.query(models.User).filter(models.User.email == email).first()
    except Exception as e:
        logger.error(f"Database query error for email {email}: {e}", exc_info=True)
        return None

def create_user(db: Session, user: schemas.UserCreate):
    try:
        db_user = models.User(
            first_name=user.first_name,
            last_name=user.last_name,
            email=user.email,
            hashed_password=hash_password(user.password),
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    except Exception as e:
        logger.error(f"Error creating user {user.email}: {e}", exc_info=True)
        db.rollback()
        raise

def authenticate_user(db: Session, email: str, password: str):
    try:
        user = get_user_by_email(db, email)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user
    except Exception as e:
        logger.error(f"Authentication error for email {email}: {e}", exc_info=True)
        return None