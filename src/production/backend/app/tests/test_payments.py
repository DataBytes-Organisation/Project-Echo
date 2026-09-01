import hashlib
import hmac
import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from app.services import payments


CAPTURED_PAYMENT = {
    "id": "pay_verified123",
    "amount": 500,
    "currency": "AUD",
    "status": "captured",
    "order_id": "order_verified123",
    "method": "card",
    "email": "donor@example.com",
    "created_at": 1700000000,
}

PAID_ORDER = {
    "id": "order_verified123",
    "amount": 500,
    "amount_paid": 500,
    "amount_due": 0,
    "currency": "AUD",
    "status": "paid",
    "notes": {"purpose": "donation"},
    "created_at": 1699999990,
}

EXPECTED_ORDER = {
    "_id": "order_verified123",
    "amount": 500,
    "currency": "AUD",
    "purpose": "donation",
    "created": 1699999990,
}

EXPECTED_DONATION = {
    "_id": "razorpay:pay_verified123",
    "paymentId": "pay_verified123",
    "orderId": "order_verified123",
    "name": "Alice",
    "email": "donor@example.com",
    "amount": 5.0,
    "currency": "aud",
    "method": "card",
    "status": "succeeded",
    "created": 1700000000,
}


class UpdateResult:
    def __init__(self, upserted_id=None):
        self.upserted_id = upserted_id


class FakeCollection:
    def __init__(self):
        self.records = {}
        self.find_calls = 0
        self.update_calls = 0
        self.fail_find = False
        self.fail_update = False
        self._lock = threading.Lock()

    def find_one(self, query):
        self.find_calls += 1
        if self.fail_find:
            raise RuntimeError("database unavailable")
        if "_id" in query:
            return self.records.get(query["_id"])
        return next(
            (record for record in self.records.values() if record.get("paymentId") == query.get("paymentId")),
            None,
        )

    def update_one(self, query, update, upsert=False):
        self.update_calls += 1
        if self.fail_update:
            raise RuntimeError("database unavailable")
        with self._lock:
            if query["_id"] in self.records:
                return UpdateResult()
            record = dict(update["$setOnInsert"])
            self.records[query["_id"]] = record
            return UpdateResult(query["_id"])


class FakeProvider:
    def __init__(self, payment=None, order=None):
        self.payment = dict(CAPTURED_PAYMENT if payment is None else payment)
        self.order = dict(PAID_ORDER if order is None else order)
        self.calls = []
        self.fail_create = False
        self.fail_get = False

    def create_order(self, body):
        self.calls.append(("create_order", body))
        if self.fail_create:
            raise payments.ProviderUnavailable()
        return dict(self.order)

    def get_payment(self, payment_id):
        self.calls.append(("get_payment", payment_id))
        if self.fail_get:
            raise payments.ProviderUnavailable()
        return dict(self.payment)

    def get_order(self, order_id):
        self.calls.append(("get_order", order_id))
        if self.fail_get:
            raise payments.ProviderUnavailable()
        return dict(self.order)


def dependencies(provider=None, orders=None, donations=None, **overrides):
    values = {
        "key_id": "rzp_test_key",
        "key_secret": "test_secret",
        "webhook_secret": "webhook_secret",
        "provider": provider or FakeProvider(),
        "orders": orders or FakeCollection(),
        "donations": donations or FakeCollection(),
    }
    values.update(overrides)
    return payments.PaymentDependencies(**values)


def checkout_signature(secret="test_secret"):
    return hmac.new(
        secret.encode(),
        b"order_verified123|pay_verified123",
        hashlib.sha256,
    ).hexdigest()


def checkout_payload(**overrides):
    payload = {
        "paymentId": "pay_verified123",
        "orderId": "order_verified123",
        "signature": checkout_signature(),
        "name": " Alice ",
    }
    payload.update(overrides)
    return payload


def retained_orders():
    orders = FakeCollection()
    orders.records[EXPECTED_ORDER["_id"]] = dict(EXPECTED_ORDER)
    return orders


def captured_webhook():
    return json.dumps(
        {
            "event": "payment.captured",
            "payload": {"payment": {"entity": CAPTURED_PAYMENT}},
        },
        separators=(",", ":"),
    ).encode()


def webhook_signature(body, secret="webhook_secret"):
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


class PaymentTests(unittest.TestCase):
    def test_create_order_rejects_non_allowed_amount_without_side_effects(self):
        for amount in (None, True, "500", 0, 101, 500.5, 10000):
            with self.subTest(amount=amount):
                provider = FakeProvider()
                orders = FakeCollection()

                result = payments.create_order(amount, dependencies(provider=provider, orders=orders))

                self.assertEqual(
                    result,
                    payments.PaymentResult(400, {"error": "Invalid donation amount."}),
                )
                self.assertEqual(provider.calls, [])
                self.assertEqual(orders.update_calls, 0)

    def test_create_order_retains_each_allowed_aud_order_before_returning(self):
        for amount in (100, 500, 1000, 2000, 5000):
            with self.subTest(amount=amount):
                provider_order = dict(PAID_ORDER, amount=amount, amount_paid=amount)
                provider = FakeProvider(order=provider_order)
                orders = FakeCollection()

                result = payments.create_order(amount, dependencies(provider=provider, orders=orders))

                self.assertEqual(result.status_code, 201)
                self.assertEqual(
                    result.body,
                    {
                        "keyId": "rzp_test_key",
                        "orderId": "order_verified123",
                        "amount": amount,
                        "currency": "AUD",
                    },
                )
                self.assertEqual(orders.records["order_verified123"]["amount"], amount)
                self.assertEqual(provider.calls[0][0], "create_order")
                self.assertRegex(
                    provider.calls[0][1]["receipt"],
                    r"^echo-donation-[a-f0-9]{24}$",
                )

    def test_create_order_reports_configuration_provider_and_database_failures_safely(self):
        provider = FakeProvider()
        result = payments.create_order(500, dependencies(provider=provider, key_secret=""))
        self.assertEqual(result, payments.PaymentResult(503, {"error": "Payment service is unavailable."}))
        self.assertEqual(provider.calls, [])

        provider.fail_create = True
        result = payments.create_order(500, dependencies(provider=provider))
        self.assertEqual(result, payments.PaymentResult(502, {"error": "Payment service is unavailable."}))

        orders = FakeCollection()
        orders.fail_update = True
        result = payments.create_order(500, dependencies(orders=orders))
        self.assertEqual(result, payments.PaymentResult(503, {"error": "Payment service is unavailable."}))

    def test_verify_checkout_rejects_malformed_and_invalid_signatures_without_side_effects(self):
        malformed = (
            {},
            checkout_payload(paymentId="../payment"),
            checkout_payload(orderId="../order"),
            checkout_payload(signature="not-a-signature"),
            checkout_payload(signature="0" * 64),
        )
        for payload in malformed:
            with self.subTest(payload=payload):
                provider = FakeProvider()
                donations = FakeCollection()
                result = payments.verify_checkout(
                    payload,
                    dependencies(provider=provider, orders=retained_orders(), donations=donations),
                )
                self.assertEqual(result.status_code, 400)
                self.assertEqual(result.body, {"error": "Payment could not be verified."})
                self.assertEqual(provider.calls, [])
                self.assertEqual(donations.update_calls, 0)

    def test_verify_checkout_rejects_unknown_order_without_provider_or_donation_calls(self):
        provider = FakeProvider()
        donations = FakeCollection()
        result = payments.verify_checkout(
            checkout_payload(),
            dependencies(provider=provider, orders=FakeCollection(), donations=donations),
        )
        self.assertEqual(result.status_code, 400)
        self.assertEqual(provider.calls, [])
        self.assertEqual(donations.update_calls, 0)

    def test_verify_checkout_rejects_each_provider_mismatch_without_writing(self):
        mutations = (
            ({"status": "authorized"}, {}),
            ({"order_id": "order_other"}, {}),
            ({"amount": 100}, {}),
            ({"currency": "INR"}, {}),
            ({"created_at": "1700000000"}, {}),
            ({}, {"status": "created"}),
            ({}, {"amount_due": 500}),
            ({}, {"amount_paid": 0}),
            ({}, {"currency": "INR"}),
            ({}, {"notes": {}}),
        )
        for payment_change, order_change in mutations:
            with self.subTest(payment=payment_change, order=order_change):
                provider = FakeProvider(
                    payment=dict(CAPTURED_PAYMENT, **payment_change),
                    order=dict(PAID_ORDER, **order_change),
                )
                donations = FakeCollection()
                result = payments.verify_checkout(
                    checkout_payload(),
                    dependencies(provider=provider, orders=retained_orders(), donations=donations),
                )
                self.assertEqual(result.status_code, 400)
                self.assertEqual(donations.update_calls, 0)

    def test_verify_checkout_stores_authoritative_fields_and_ignores_forged_financial_fields(self):
        donations = FakeCollection()
        payload = checkout_payload(
            amount=999999,
            currency="USD",
            email="forged@example.com",
            status="forged",
        )

        result = payments.verify_checkout(
            payload,
            dependencies(orders=retained_orders(), donations=donations),
        )

        self.assertEqual(result, payments.PaymentResult(201, {"status": "success"}))
        self.assertEqual(donations.records[EXPECTED_DONATION["_id"]], EXPECTED_DONATION)

    def test_verify_checkout_treats_historical_and_concurrent_replay_as_success(self):
        historical = FakeCollection()
        historical.records["legacy-id"] = {"_id": "legacy-id", "paymentId": "pay_verified123"}
        provider = FakeProvider()
        result = payments.verify_checkout(
            checkout_payload(),
            dependencies(provider=provider, orders=retained_orders(), donations=historical),
        )
        self.assertEqual(result, payments.PaymentResult(200, {"status": "success"}))
        self.assertEqual(provider.calls, [])

        concurrent = FakeCollection()
        deps = dependencies(orders=retained_orders(), donations=concurrent)
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: payments.verify_checkout(checkout_payload(), deps), range(2)))
        self.assertEqual(sorted(result.status_code for result in results), [200, 201])
        self.assertEqual(list(concurrent.records), ["razorpay:pay_verified123"])

    def test_verify_checkout_reports_provider_and_database_failures_without_success(self):
        provider = FakeProvider()
        provider.fail_get = True
        result = payments.verify_checkout(
            checkout_payload(),
            dependencies(provider=provider, orders=retained_orders()),
        )
        self.assertEqual(result, payments.PaymentResult(502, {"error": "Payment service is unavailable."}))

        for failure in ("find", "update"):
            with self.subTest(failure=failure):
                donations = FakeCollection()
                setattr(donations, f"fail_{failure}", True)
                result = payments.verify_checkout(
                    checkout_payload(),
                    dependencies(orders=retained_orders(), donations=donations),
                )
                self.assertEqual(result, payments.PaymentResult(503, {"error": "Donation could not be recorded."}))

    def test_webhook_requires_matching_hmac_over_exact_bytes(self):
        body = captured_webhook()
        for signature in (None, "not-a-signature", "0" * 64):
            with self.subTest(signature=signature):
                provider = FakeProvider()
                result = payments.process_webhook(body, signature, dependencies(provider=provider))
                self.assertEqual(result, payments.PaymentResult(400, {"error": "Invalid webhook."}))
                self.assertEqual(provider.calls, [])

        changed_body = body + b" "
        result = payments.process_webhook(
            changed_body,
            webhook_signature(body),
            dependencies(),
        )
        self.assertEqual(result, payments.PaymentResult(400, {"error": "Invalid webhook."}))

    def test_webhook_rejects_malformed_json_and_identifiers(self):
        for body in (
            b'{"event":"payment.captured"',
            json.dumps(
                {
                    "event": "payment.captured",
                    "payload": {"payment": {"entity": {"id": "../payment", "order_id": "../order"}}},
                }
            ).encode(),
        ):
            with self.subTest(body=body):
                result = payments.process_webhook(body, webhook_signature(body), dependencies())
                self.assertEqual(result, payments.PaymentResult(400, {"error": "Invalid webhook."}))

    def test_webhook_ignores_unrelated_signed_event_without_side_effects(self):
        body = b'{  "event": "order.paid"  }'
        provider = FakeProvider()
        donations = FakeCollection()
        result = payments.process_webhook(
            body,
            webhook_signature(body),
            dependencies(provider=provider, donations=donations),
        )
        self.assertEqual(result, payments.PaymentResult(200, {"status": "ignored"}))
        self.assertEqual(provider.calls, [])
        self.assertEqual(donations.update_calls, 0)

    def test_webhook_uses_the_same_idempotent_capture_path(self):
        body = captured_webhook()
        donations = FakeCollection()
        result = payments.process_webhook(
            body,
            webhook_signature(body),
            dependencies(orders=retained_orders(), donations=donations),
        )
        self.assertEqual(result, payments.PaymentResult(201, {"status": "success"}))
        expected = dict(EXPECTED_DONATION, name="Anonymous")
        self.assertEqual(donations.records[EXPECTED_DONATION["_id"]], expected)


if __name__ == "__main__":
    unittest.main()
