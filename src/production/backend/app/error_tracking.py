import os
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration


def init_error_tracking():
    dsn = os.getenv("SENTRY_DSN")

    if not dsn:
        print("Sentry error tracking disabled: SENTRY_DSN is not configured")
        return

    sentry_sdk.init(
        dsn=dsn,
        environment=os.getenv("SENTRY_ENVIRONMENT", "development"),
        integrations=[FastApiIntegration()],
        send_default_pii=False,
        traces_sample_rate=0.0,
    )

    print("Sentry error tracking enabled")


def capture_critical_error(error, operation=None, context=None):
    """
    Capture critical operational errors that are handled by the application
    but still need to be reported to Sentry.
    """
    with sentry_sdk.push_scope() as scope:
        scope.set_tag("severity", "critical")

        if operation:
            scope.set_tag("operation", operation)

        if context:
            scope.set_context("project_echo", context)

        sentry_sdk.capture_exception(error)
