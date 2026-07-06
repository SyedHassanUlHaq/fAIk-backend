from pydantic import BaseModel


class SubscribeRequest(BaseModel):
    planId: str
    paymentMethodId: str
