from fastapi import APIRouter, HTTPException
import logging
import stripe
from schemas.schemas import PaymentIntentCreate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/payments", tags=["Payments"])

#----------------------------------------------------------------------------------------
# Stripe Configuration
#----------------------------------------------------------------------------------------

@router.post("/create-payment-intent")
def create_payment_intent(payload: PaymentIntentCreate):
    try:
        try:
            intent = stripe.PaymentIntent.create(
                amount=payload.amount,
                currency="usd",
                automatic_payment_methods={"enabled": True},
                metadata={
                    "user_id": payload.user_id,
                    "order_id": payload.order_id
                }
            )
            return {"client_secret": intent.client_secret}
        except stripe.error.CardError as e:
            logger.warning(f"Card error creating payment intent: {e.user_message}")
            raise HTTPException(status_code=400, detail=e.user_message)
        except stripe.error.RateLimitError:
            logger.error("Stripe rate limit exceeded")
            raise HTTPException(status_code=429, detail="Service temporarily unavailable")
        except stripe.error.AuthenticationError:
            logger.error("Stripe authentication failed")
            raise HTTPException(status_code=500, detail="Payment service configuration error")
        except stripe.error.APIConnectionError:
            logger.error("Stripe API connection failed")
            raise HTTPException(status_code=503, detail="Payment service unavailable")
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error creating payment intent: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail="Failed to create payment intent")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating payment intent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create payment intent")