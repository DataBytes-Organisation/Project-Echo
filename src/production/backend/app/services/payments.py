import base64
import hashlib
import hmac
import json
import re
import secrets
import urllib.request
from dataclasses import dataclass


API_URL = "https://api.razorpay.com/v1"
ALLOWED_AMOUNTS = frozenset({100, 500, 1000, 2000, 5000})
PAYMENT_ID = re.compile(r"^pay_[A-Za-z0-9]+$")
ORDER_ID = re.compile(r"^order_[A-Za-z0-9]+$")
SIGNATURE = re.compile(r"^[a-fA-F0-9]{64}$")

PAYMENT_UNAVAILABLE = {"error": "Payment service is unavailable."}
PAYMENT_UNVERIFIED = {"error": "Payment could not be verified."}
DONATION_UNRECORDED = {"error": "Donation could not be recorded."}
INVALID_WEBHOOK = {"error": "Invalid webhook."}


@dataclass(frozen=True)
class PaymentResult:
    status_code: int
    body: dict


@dataclass(frozen=True)
class PaymentDependencies:
    key_id: str
    key_secret: str
    webhook_secret: str
    orders: object
    donations: object
    provider: object = None


class ProviderUnavailable(Exception):
    pass


class RazorpayClient:
    def __init__(self, key_id, key_secret):
        token = base64.b64encode(f"{key_id}:{key_secret}".encode()).decode()
        self.authorization = f"Basic {token}"

    def _request(self, method, path, body=None):
        encoded_body = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            f"{API_URL}{path}",
            data=encoded_body,
            method=method,
            headers={
                "Authorization": self.authorization,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode())
        except Exception as error:
            raise ProviderUnavailable() from error

    def create_order(self, body):
        return self._request("POST", "/orders", body)

    def get_payment(self, payment_id):
        return self._request("GET", f"/payments/{payment_id}")

    def get_order(self, order_id):
        return self._request("GET", f"/orders/{order_id}")


def _valid_text(value):
    return isinstance(value, str) and bool(value.strip())


def _provider(dependencies):
    return dependencies.provider or RazorpayClient(
        dependencies.key_id,
        dependencies.key_secret,
    )


def create_order(amount, dependencies):
    if type(amount) is not int or amount not in ALLOWED_AMOUNTS:
        return PaymentResult(400, {"error": "Invalid donation amount."})
    if not _valid_text(dependencies.key_id) or not _valid_text(dependencies.key_secret):
        return PaymentResult(503, PAYMENT_UNAVAILABLE)
    if dependencies.orders is None:
        return PaymentResult(503, PAYMENT_UNAVAILABLE)

    request_body = {
        "amount": amount,
        "currency": "AUD",
        "receipt": f"echo-donation-{secrets.token_hex(12)}",
        "notes": {"purpose": "donation"},
    }
    try:
        order = _provider(dependencies).create_order(request_body)
    except ProviderUnavailable:
        return PaymentResult(502, PAYMENT_UNAVAILABLE)

    if (
        not isinstance(order, dict)
        or not ORDER_ID.fullmatch(str(order.get("id", "")))
        or order.get("amount") != amount
        or order.get("currency") != "AUD"
    ):
        return PaymentResult(502, PAYMENT_UNAVAILABLE)

    stored_order = {
        "_id": order["id"],
        "amount": order["amount"],
        "currency": order["currency"],
        "purpose": "donation",
        "created": order.get("created_at"),
    }
    try:
        dependencies.orders.update_one(
            {"_id": stored_order["_id"]},
            {"$setOnInsert": stored_order},
            upsert=True,
        )
    except Exception:
        return PaymentResult(503, PAYMENT_UNAVAILABLE)

    return PaymentResult(
        201,
        {
            "keyId": dependencies.key_id,
            "orderId": order["id"],
            "amount": order["amount"],
            "currency": order["currency"],
        },
    )


def _valid_hmac(message, signature, secret):
    if not isinstance(signature, str) or not SIGNATURE.fullmatch(signature):
        return False
    expected = hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature.lower())


def _verified_provider_records(payment, order, expected_order, payment_id):
    notes = order.get("notes") if isinstance(order, dict) else None
    return (
        isinstance(payment, dict)
        and isinstance(order, dict)
        and isinstance(expected_order, dict)
        and payment.get("id") == payment_id
        and order.get("id") == expected_order.get("_id")
        and payment.get("order_id") == expected_order.get("_id")
        and payment.get("status") == "captured"
        and order.get("status") == "paid"
        and isinstance(notes, dict)
        and notes.get("purpose") == "donation"
        and expected_order.get("purpose") == "donation"
        and payment.get("currency") == "AUD"
        and order.get("currency") == "AUD"
        and expected_order.get("currency") == "AUD"
        and payment.get("amount") == order.get("amount")
        and payment.get("amount") == expected_order.get("amount")
        and payment.get("amount") in ALLOWED_AMOUNTS
        and order.get("amount_paid") == order.get("amount")
        and order.get("amount_due") == 0
        and type(payment.get("created_at")) is int
    )


def _capture(payment_id, order_id, name, dependencies):
    if not _valid_text(dependencies.key_id) or not _valid_text(dependencies.key_secret):
        return PaymentResult(503, PAYMENT_UNAVAILABLE)
    if not PAYMENT_ID.fullmatch(payment_id or "") or not ORDER_ID.fullmatch(order_id or ""):
        return PaymentResult(400, PAYMENT_UNVERIFIED)
    if dependencies.orders is None or dependencies.donations is None:
        return PaymentResult(503, DONATION_UNRECORDED)

    try:
        expected_order = dependencies.orders.find_one({"_id": order_id})
    except Exception:
        return PaymentResult(503, DONATION_UNRECORDED)
    if expected_order is None:
        return PaymentResult(400, PAYMENT_UNVERIFIED)

    try:
        existing = dependencies.donations.find_one({"paymentId": payment_id})
    except Exception:
        return PaymentResult(503, DONATION_UNRECORDED)
    if existing is not None:
        return PaymentResult(200, {"status": "success"})

    provider = _provider(dependencies)
    try:
        payment = provider.get_payment(payment_id)
        order = provider.get_order(order_id)
    except ProviderUnavailable:
        return PaymentResult(502, PAYMENT_UNAVAILABLE)

    if not _verified_provider_records(payment, order, expected_order, payment_id):
        return PaymentResult(400, PAYMENT_UNVERIFIED)

    donor_name = name.strip()[:100] if _valid_text(name) else "Anonymous"
    email = payment.get("email") if _valid_text(payment.get("email")) else "N/A"
    method = payment.get("method") if _valid_text(payment.get("method")) else "unknown"
    donation = {
        "_id": f"razorpay:{payment['id']}",
        "paymentId": payment["id"],
        "orderId": order["id"],
        "name": donor_name,
        "email": email,
        "amount": payment["amount"] / 100,
        "currency": payment["currency"].lower(),
        "method": method,
        "status": "succeeded",
        "created": payment["created_at"],
    }
    try:
        result = dependencies.donations.update_one(
            {"_id": donation["_id"]},
            {"$setOnInsert": donation},
            upsert=True,
        )
    except Exception:
        return PaymentResult(503, DONATION_UNRECORDED)

    status_code = 201 if getattr(result, "upserted_id", None) is not None else 200
    return PaymentResult(status_code, {"status": "success"})


def verify_checkout(payload, dependencies):
    payload = payload if isinstance(payload, dict) else {}
    payment_id = payload.get("paymentId")
    order_id = payload.get("orderId")
    signature = payload.get("signature")
    if not _valid_text(dependencies.key_id) or not _valid_text(dependencies.key_secret):
        return PaymentResult(503, PAYMENT_UNAVAILABLE)
    if (
        not isinstance(payment_id, str)
        or not isinstance(order_id, str)
        or not _valid_hmac(
            f"{order_id}|{payment_id}".encode(),
            signature,
            dependencies.key_secret,
        )
    ):
        return PaymentResult(400, PAYMENT_UNVERIFIED)
    return _capture(payment_id, order_id, payload.get("name"), dependencies)


def process_webhook(raw_body, signature, dependencies):
    if not _valid_text(dependencies.webhook_secret):
        return PaymentResult(503, PAYMENT_UNAVAILABLE)
    if not isinstance(raw_body, bytes) or not _valid_hmac(
        raw_body,
        signature,
        dependencies.webhook_secret,
    ):
        return PaymentResult(400, INVALID_WEBHOOK)
    try:
        event = json.loads(raw_body.decode())
    except (UnicodeDecodeError, json.JSONDecodeError):
        return PaymentResult(400, INVALID_WEBHOOK)
    if not isinstance(event, dict):
        return PaymentResult(400, INVALID_WEBHOOK)
    if event.get("event") != "payment.captured":
        return PaymentResult(200, {"status": "ignored"})

    payload = event.get("payload")
    payment_payload = payload.get("payment") if isinstance(payload, dict) else None
    payment = payment_payload.get("entity") if isinstance(payment_payload, dict) else None
    payment_id = payment.get("id") if isinstance(payment, dict) else None
    order_id = payment.get("order_id") if isinstance(payment, dict) else None
    if (
        not isinstance(payment_id, str)
        or not PAYMENT_ID.fullmatch(payment_id)
        or not isinstance(order_id, str)
        or not ORDER_ID.fullmatch(order_id)
    ):
        return PaymentResult(400, INVALID_WEBHOOK)
    return _capture(payment_id, order_id, "Anonymous", dependencies)
