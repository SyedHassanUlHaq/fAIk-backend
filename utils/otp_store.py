from datetime import datetime, timedelta
import logging
from typing import Dict
from config.project_config import OTP_EXPIRE_MINUTES

logger = logging.getLogger(__name__)

otp_store: Dict[str, dict] = {}
# Structure:
# otp_store[email] = {
#     "otp": "123456",
#     "data": {first_name, last_name, email, password},
#     "expires_at": datetime
# }

def store_otp(email: str, otp: str, user_data: dict):
    try:
        otp_store[email] = {
            "otp": otp,
            "data": user_data,
            "expires_at": datetime.utcnow() + timedelta(minutes=OTP_EXPIRE_MINUTES)
        }
    except Exception as e:
        logger.error(f"Error storing OTP for {email}: {e}", exc_info=True)
        raise

def verify_otp(email: str, otp: str):
    try:
        record = otp_store.get(email)
        if not record:
            return False, "No OTP found for this email"
        if datetime.utcnow() > record["expires_at"]:
            del otp_store[email]
            return False, "OTP expired"
        if record["otp"] != otp:
            return False, "Invalid OTP"
        return True, record["data"]
    except Exception as e:
        logger.error(f"Error verifying OTP for {email}: {e}", exc_info=True)
        return False, "Error verifying OTP"

def delete_otp(email: str):
    try:
        if email in otp_store:
            del otp_store[email]
    except Exception as e:
        logger.error(f"Error deleting OTP for {email}: {e}", exc_info=True)