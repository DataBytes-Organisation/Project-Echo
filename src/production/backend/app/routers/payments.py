import os

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.database import Donations, RazorpayOrders
from app.middleware.auth_bearer import JWTBearer
from app.schemas import PaymentStatusResponse, RazorpayCheckoutProof, RazorpayOrderRequest, RazorpayOrderResponse
from app.services import payments as payment_service


router = APIRouter(prefix="/payments/razorpay", tags=["payments"])
jwt_bearer = JWTBearer()


def _dependencies():
    return payment_service.PaymentDependencies(
        key_id=os.getenv("RAZORPAY_KEY_ID", ""),
        key_secret=os.getenv("RAZORPAY_KEY_SECRET", ""),
        webhook_secret=os.getenv("RAZORPAY_WEBHOOK_SECRET", ""),
        orders=RazorpayOrders,
        donations=Donations,
    )


def _response(result):
    return JSONResponse(status_code=result.status_code, content=result.body)


@router.post(
    "/orders",
    response_model=RazorpayOrderResponse,
    status_code=201,
    dependencies=[Depends(jwt_bearer)],
)
def create_razorpay_order(payload: RazorpayOrderRequest):
    return _response(payment_service.create_order(payload.amount, _dependencies()))


@router.post("/verify", response_model=PaymentStatusResponse)
def verify_razorpay_checkout(payload: RazorpayCheckoutProof):
    return _response(payment_service.verify_checkout(payload.dict(), _dependencies()))


@router.post("/webhook", response_model=PaymentStatusResponse)
async def receive_razorpay_webhook(request: Request):
    raw_body = await request.body()
    signature = request.headers.get("x-razorpay-signature")
    result = await run_in_threadpool(
        payment_service.process_webhook,
        raw_body,
        signature,
        _dependencies(),
    )
    return _response(result)
