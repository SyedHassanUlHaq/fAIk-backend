import os
import stripe
from fastapi import APIRouter, Request, HTTPException, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Payment

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])

@router.post("/stripe")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, os.getenv("STRIPE_WEBHOOK_SECRET")
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Stripe signature")

    if event["type"] == "payment_intent.succeeded":
        intent = event["data"]["object"]

        user_id = intent.metadata.get("user_id")
        order_id = intent.metadata.get("order_id")

        existing = db.query(Payment).filter_by(
            stripe_payment_intent_id=intent.id
        ).first()

        if not existing:
            payment = Payment(
                user_id=int(user_id),
                stripe_payment_intent_id=intent.id,
                amount=intent.amount,
                status=intent.status,
            )
            db.add(payment)
            db.commit()

    return {"status": "ok"}