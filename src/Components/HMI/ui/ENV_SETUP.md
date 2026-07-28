# Environment Setup Guide

This guide explains how to configure the `.env` file for the HMI service so it works correctly both locally and on the live server.

## What is `.env`?

The `.env` file holds environment-specific settings such as the website address (`CLIENT_URL`) and the backend API host (`API_HOST`). The same code is used everywhere — only the `.env` file changes between local and live environments.

## Initial Setup

1. Navigate to the HMI UI folder: `src/Components/HMI/ui/`
2. Create a new file named `.env` (copy from `.env.example` if it exists).
3. Add the following variables:

```
API_HOST=localhost
CLIENT_URL=http://localhost:8080
```

## Variable Descriptions

- `API_HOST` — Backend API host. Locally set to `localhost`. Automatically switches to `api-service` in Kubernetes via the ConfigMap.
- `CLIENT_URL` — Frontend URL used for Stripe redirects and password reset emails. Locally set to `http://localhost:8080`. On the live server, this is provided by the Cloud team.

## Updating `CLIENT_URL` for Live Server

When the Cloud team provides the live server URL (e.g. `http://4.147.145.111:8080`), update the `.env` file on the live server only:

```
CLIENT_URL=http://4.147.145.111:8080
```

Then restart the server so the new value takes effect.

## Verification

After updating, the connection status badge on the admin dashboard will display:
- `Running in Local Mode — API connected successfully.` when running locally.
- `Running in Live Mode — API connected successfully.` when running on the live server.
- `Live server URL not configured yet, running in local fallback mode.` if `CLIENT_URL` is empty or misconfigured.

## Troubleshooting

- **Badge shows "Cannot connect to backend API"** → The backend API service is not running or unreachable. Check that all Docker containers are up.
- **Badge shows "Fallback Mode"** → `CLIENT_URL` is not set correctly in the `.env` file.
- **Changes to `.env` not taking effect** → Restart Docker with `docker compose down && docker compose up --build`.