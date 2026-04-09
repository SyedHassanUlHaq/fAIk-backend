import os
import logging
import stripe
from fastapi import APIRouter, Request, HTTPException, Depends
from sqlalchemy.orm import Session
from database import get_db
from models.payments import Payment

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])

#----------------------------------------------------------------------------------------
# Stripe Webhook Endpoint
#----------------------------------------------------------------------------------------

@router.post("/stripe")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    try:
        payload = await request.body()
        sig_header = request.headers.get("stripe-signature")

        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, os.getenv("STRIPE_WEBHOOK_SECRET")
            )
        except stripe.error.SignatureVerificationError as e:
            logger.warning(f"Invalid Stripe signature: {e}")
            raise HTTPException(status_code=400, detail="Invalid Stripe signature")
        except Exception as e:
            logger.error(f"Error constructing Stripe event: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail="Failed to process webhook")

        if event["type"] == "payment_intent.succeeded":
            try:
                intent = event["data"]["object"]

                user_id = intent.metadata.get("user_id")
                order_id = intent.metadata.get("order_id")

                if not user_id:
                    logger.warning(f"Missing user_id in payment intent {intent.id}")
                    return {"status": "ok"}

                existing = db.query(Payment).filter_by(
                    stripe_payment_intent_id=intent.id
                ).first()

                if not existing:
                    try:
                        payment = Payment(
                            user_id=int(user_id),
                            stripe_payment_intent_id=intent.id,
                            amount=intent.amount,
                            status=intent.status,
                        )
                        db.add(payment)
                        db.commit()
                    except ValueError as e:
                        logger.error(f"Invalid user_id format: {user_id}, error: {e}", exc_info=True)
                        db.rollback()
                    except Exception as e:
                        logger.error(f"Error saving payment for intent {intent.id}: {e}", exc_info=True)
                        db.rollback()
            except KeyError as e:
                logger.error(f"Missing expected key in webhook event: {e}", exc_info=True)
        
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in stripe_webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Webhook processing failed")