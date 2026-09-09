# Backend Environment Setup Guide

This guide explains how to configure the Backend `.env` file for local and deployed environments.

## Initial Setup

1. Navigate to `src/production/backend/`.
2. Create `.env` from `.env.example`.
3. Set the service variables required for your environment. For Razorpay payments, set:

```
RAZORPAY_KEY_ID=
RAZORPAY_KEY_SECRET=
RAZORPAY_WEBHOOK_SECRET=
```

Use a matching Razorpay Test Mode or Live Mode key pair. Obtain the webhook secret from the Razorpay webhook configuration for the Backend's `/payments/razorpay/webhook` endpoint.

## Razorpay Variables

- `RAZORPAY_KEY_ID` — Identifies the Razorpay account and is returned only as part of a Backend-created checkout order.
- `RAZORPAY_KEY_SECRET` — Authenticates Backend requests to Razorpay. Keep it server-side and never commit or expose it to browser code.
- `RAZORPAY_WEBHOOK_SECRET` — Verifies incoming Razorpay webhook signatures. Keep it server-side and never commit it.

The HMI only proxies payment requests to the Backend; it does not require Razorpay credentials in its `.env` file.

## Troubleshooting

- **Payment service is unavailable** — Confirm all three Razorpay variables are set in the Backend environment, use matching Test or Live Mode credentials, and restart the Backend service.
- **Webhook verification fails** — Check that `RAZORPAY_WEBHOOK_SECRET` matches the secret configured for the Backend webhook endpoint.
