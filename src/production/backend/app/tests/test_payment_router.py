import sys
import types
import unittest
import warnings
from unittest.mock import patch


fake_database = types.ModuleType("app.database")
fake_database.GENDER = []
fake_database.STATES_CODE = []
fake_database.AUS_STATES = []
fake_database.Donations = object()
fake_database.RazorpayOrders = object()
original_database = sys.modules.get("app.database")
sys.modules["app.database"] = fake_database

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import payments as payment_router
from app.services.payments import PaymentResult

if original_database is None:
    sys.modules.pop("app.database", None)
else:
    sys.modules["app.database"] = original_database


CHECKOUT_ORDER = {
    "keyId": "rzp_test_key",
    "orderId": "order_verified123",
    "amount": 500,
    "currency": "AUD",
}


class PaymentRouterTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(payment_router.router)
        self.app = app
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The 'app' shortcut is now deprecated.*")
            self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def test_order_endpoint_requires_bearer_authentication(self):
        response = self.client.post(
            "/payments/razorpay/orders",
            json={"amount": 500},
        )
        self.assertEqual(response.status_code, 403)

    def test_order_endpoint_returns_payment_service_status_and_body(self):
        self.app.dependency_overrides[payment_router.jwt_bearer] = lambda: "requester-jwt"
        with patch.object(
            payment_router.payment_service,
            "create_order",
            return_value=PaymentResult(201, CHECKOUT_ORDER),
        ) as create_order:
            response = self.client.post(
                "/payments/razorpay/orders",
                json={"amount": 500},
            )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json(), CHECKOUT_ORDER)
        self.assertEqual(create_order.call_args.args[0], 500)

    def test_verify_endpoint_rejects_incomplete_proof_before_service_call(self):
        with patch.object(payment_router.payment_service, "verify_checkout") as verify_checkout:
            response = self.client.post(
                "/payments/razorpay/verify",
                json={"paymentId": "pay_verified123"},
            )

        self.assertEqual(response.status_code, 422)
        verify_checkout.assert_not_called()

    def test_verify_endpoint_passes_typed_checkout_proof(self):
        proof = {
            "paymentId": "pay_verified123",
            "orderId": "order_verified123",
            "signature": "a" * 64,
            "name": "Alice",
        }
        with patch.object(
            payment_router.payment_service,
            "verify_checkout",
            return_value=PaymentResult(200, {"status": "success"}),
        ) as verify_checkout:
            response = self.client.post("/payments/razorpay/verify", json=proof)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "success"})
        self.assertEqual(verify_checkout.call_args.args[0], proof)

    def test_webhook_endpoint_passes_exact_body_and_signature_to_service(self):
        raw_body = b'{  "event": "order.paid"  }'
        with patch.object(
            payment_router.payment_service,
            "process_webhook",
            return_value=PaymentResult(200, {"status": "ignored"}),
        ) as process_webhook:
            response = self.client.post(
                "/payments/razorpay/webhook",
                content=raw_body,
                headers={"x-razorpay-signature": "a" * 64},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ignored"})
        self.assertEqual(process_webhook.call_args.args[0], raw_body)
        self.assertEqual(process_webhook.call_args.args[1], "a" * 64)

    def test_router_exposes_all_three_payment_paths(self):
        paths = self.app.openapi()["paths"]
        self.assertIn("/payments/razorpay/orders", paths)
        self.assertIn("/payments/razorpay/verify", paths)
        self.assertIn("/payments/razorpay/webhook", paths)
        self.assertIn("201", paths["/payments/razorpay/orders"]["post"]["responses"])
        self.assertEqual(
            paths["/payments/razorpay/verify"]["post"]["responses"]["200"]["content"]["application/json"]["schema"],
            {"$ref": "#/components/schemas/PaymentStatusResponse"},
        )


if __name__ == "__main__":
    unittest.main()
