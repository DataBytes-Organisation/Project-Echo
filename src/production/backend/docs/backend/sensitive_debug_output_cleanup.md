# Sensitive Debug Output Cleanup

## Summary

This task removes or sanitises sensitive debug output from selected backend authentication, SMS, and MQTT paths while preserving existing application behaviour.

The cleanup focuses on preventing credentials, OTPs, JWTs, phone numbers, request data, raw MQTT payloads, and exception details from being unnecessarily exposed through console or application logs.

## Scope Reviewed

The following files were reviewed:

- `app/middleware/auth.py`
- `app/middleware/auth_bearer.py`
- `app/routers/auth_router.py`
- `app/utils/sms.py`
- `app/services/mqtt_client.py`

## Changes

### SMS logging

Updated:

`app/utils/sms.py`

The SMS failure path previously logged exception traceback/details using:

logger.warning("Error sending SMS", exc_info=True)

This was replaced with a generic operational message:

logger.warning("SMS sending failed")

The existing return behaviour remains unchanged.

## MQTT logging

### Updated:

app/services/mqtt_client.py

Direct print() debugging was replaced with structured logging.

### The following sensitive or unnecessarily detailed output was removed:

full normalised MQTT event objects;
raw exception messages from initial MQTT connection failures;
payload key lists for unrecognised MQTT messages;
direct console output for connection events.

### Safe operational context is retained, including:

MQTT connection state;
MQTT return code;
event type;
generic malformed/unrecognised payload warnings.

### The existing MQTT behaviour remains unchanged, including:

subscriptions;
connection state transitions;
event normalisation;
latest_events storage;
fallback raw event handling.

## Authentication Review

### No code changes were required in:

app/middleware/auth.py
app/middleware/auth_bearer.py
app/routers/auth_router.py

### The current authentication logs do not output:

passwords;
generated OTP values;
JWT values;
decoded JWT payloads;
complete user records.

### For example, OTP generation records only:

OTP generated for sign-in request

without including the OTP or user email.

## Verification
### Authentication regression test

Command:

python -m pytest -q tests/test_engine_auth.py

Result:

4 passed

One existing python_multipart deprecation warning was reported and is unrelated to this task.

## Authentication / OTP smoke test

Verified that:

valid credentials still initiate OTP generation;
OTP is stored correctly;
OTP verification still succeeds;
OTP is removed after successful verification;
JWT generation/return behaviour remains intact;
email, password, OTP and token values are not written to logs.

Result:

signin_response_ok: True
otp_stored: True
verify_response_ok: True
otp_removed_after_verification: True
sensitive_values_not_logged: True
captured_log: 'OTP generated for sign-in request'

## MQTT smoke tests

Verified:

malformed JSON is rejected without logging the raw payload;
unknown payloads preserve the existing fallback behaviour;
on_message() continues storing normalised events in latest_events;
connection and reconnection state behaviour remains unchanged;
MQTT subscriptions remain unchanged.

Results:

unknown_payload: True
event_stored: True
connected_state: True
subscriptions_ok: True
reconnecting_state: True

## SMS failure-path test

A mocked SMS exception containing a unique sensitive marker was used.

Result:

SMS sending failed
sms_failure_returns_false: True

The exception marker, recipient phone number, and message contents were not emitted to the logs.

## Sensitive print audit

The five scoped files were searched for active print() calls.

Result:

No matches

## Diff validation
git diff --check

completed without whitespace errors.

## Full Test Suite Note

The complete repository test suite could not complete collection because of unrelated existing test-environment/repository issues:

httpx is not installed for several FastAPI / Starlette TestClient tests;
several tests currently import Events from app.database, but that symbol is unavailable in the current main state.

These issues are outside the scope of this cleanup and were not modified.

## Out of Scope Finding

Additional debug print() statements were observed elsewhere in the backend during the initial audit, including app/routers/hmi.py.

These were intentionally not modified because they are outside the defined scope of this task. They should be considered separately for a broader logging/security cleanup.

## Conclusion

Sensitive debug output in the defined task scope has been removed or sanitised without changing authentication, SMS, or MQTT functional behaviour.