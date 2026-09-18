# Environment Setup Guide

The HMI reads local settings from `.env` in `src/production/hmi/ui/`. Copy
`.env.example` to `.env` when you need values that differ from the defaults.

## Local Values

```env
API_HOST=localhost
API_PORT=9000
CLIENT_URL=http://localhost:3000
COOKIE_SECRET=replace-with-a-long-random-value
```

The Express HMI listens on `http://localhost:3000`. `API_HOST` and `API_PORT`
identify the FastAPI Backend that the HMI proxies to. `CLIENT_URL` is the public
browser-facing HMI URL used by redirects and email links.

## Runtime Notes

- `COOKIE_SECRET` should be stable and random anywhere sessions must survive a restart.
- Docker supplies service hostnames such as the Backend, Redis, and MongoDB hosts through the container environment.
- The HMI Docker image installs dependencies with `npm ci --omit=dev` and starts `node server.js`.
- Restart the HMI process after changing `.env`.

## Verification

From `src/production/hmi/ui/`:

```powershell
npm ci
npm test
node server.js
```

Then open `http://localhost:3000/login`.

## Troubleshooting

- **Cannot connect to Backend** - confirm the Backend is running and `API_HOST`/`API_PORT` point to it.
- **Sessions reset after restart** - set a stable `COOKIE_SECRET`.
- **Payment checkout is unavailable** - confirm the Backend payment provider environment is configured, then restart the Backend and HMI.
- **Changes to `.env` do not appear** - restart the HMI process or rebuild/restart the container.
