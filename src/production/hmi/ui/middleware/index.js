"use strict";

/**
 * index.js
 * Session middleware and Redis client for the EchoNet backend.
 *
 * Sprint 1/2 : Redis JWT session check, checkUserSession middleware.
 * Task 7     : Fixed inverted / broken route guard logic.
 *              Removed dead null/undefined checks on req.path.
 *              Separated public routes (no token needed) from protected routes
 *              (token required) so the intent is explicit and easy to extend.
 *              Added clearUserSession() for logout flows.
 *              Improved Redis error handling and connection guard.
 */

const verifySignUp = require("./verifySignup");
const redis = require("redis");
const { createCheckUserSession } = require("./session");

// ─────────────────────────────────────────────────────────────────────────────
// Redis client
// ─────────────────────────────────────────────────────────────────────────────

const client = redis.createClient({
  socket: {
    host: process.env.REDIS_HOST || "echo-redis",
    port: parseInt(process.env.REDIS_PORT || "6379", 10),
  },
});

client.on("error", (err) => {
  console.error("Redis client error:", err);
});

/**
 * Ensure the Redis client is connected before use.
 * Guards against both a closed connection and a not-yet-ready state.
 */
async function ensureRedisConnected() {
  if (!client.isOpen) {
    await client.connect();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Route lists
//
// PUBLIC_ROUTES  — accessible without a session token.
// All other routes require the request's cookie-session JWT to match Redis.
//
// To add a new public route, add its exact path string to PUBLIC_ROUTES or
// its prefix to PUBLIC_PREFIXES below.  Do not add it to the middleware
// condition directly — keeping the lists here makes auditing straightforward.
// ─────────────────────────────────────────────────────────────────────────────

const PUBLIC_ROUTES = new Set(["/login", "/signup", "/map"]);

/**
 * Path prefixes that are always public regardless of the full path.
 * e.g. "/admin" covers "/admin", "/admin/users", "/admin/settings".
 */
const PUBLIC_PREFIXES = ["/admin", "/public", "/static"];

/**
 * Return true if the given path should be accessible without a session token.
 *
 * @param {string} path - Express req.path value.
 * @returns {boolean}
 */
function _isPublicRoute(path) {
  if (PUBLIC_ROUTES.has(path)) return true;
  return PUBLIC_PREFIXES.some((prefix) => path.startsWith(prefix));
}

// ─────────────────────────────────────────────────────────────────────────────
// Session middleware
// ─────────────────────────────────────────────────────────────────────────────

const checkUserSession = createCheckUserSession(client, _isPublicRoute);

// ─────────────────────────────────────────────────────────────────────────────
// API session guard
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build a session guard for routes that answer with JSON.
 *
 * checkUserSession above redirects to /login when there is no session, which is
 * right for a page but wrong for an API: axios follows the redirect, gets the
 * login page back with a 200, and the caller ends up parsing HTML as JSON. That
 * is the same failure that had to be fixed in the sign-in route. This replies
 * with a status the caller can actually act on instead.
 *
 * The token lookup is passed in so the guard can be tested without Redis.
 *
 * @param {() => Promise<string|null>} getToken
 * @returns {import('express').RequestHandler}
 */
function createApiSessionGuard(getToken) {
  return async function apiSessionGuard(req, res, next) {
    try {
      const token = await getToken();

      if (!token) {
        return res.status(401).json({
          error: "Your session has expired. Please sign in again."
        });
      }

      return next();
    } catch (error) {
      // Redis being unreachable is our problem rather than the caller's, so this
      // is a 503 and not a 401. A 401 would tell them to sign in again, which
      // would not help and would lose whatever they were doing.
      console.error("API session check failed (Redis error):", error);
      return res.status(503).json({
        error: "Unable to verify your session. Please try again shortly."
      });
    }
  };
}

/**
 * Session guard for JSON API routes, backed by the same Redis JWT that
 * checkUserSession reads.
 */
const requireApiSession = createApiSessionGuard(async () => {
  await ensureRedisConnected();
  return client.get("JWT");
});

// ─────────────────────────────────────────────────────────────────────────────
// Session helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Remove the JWT from Redis, effectively logging the user out.
 * Call this from your logout route handler.
 *
 * @returns {Promise<void>}
 *
 * @example
 * app.post("/logout", async (req, res) => {
 *   await clearUserSession();
 *   res.redirect("/login");
 * });
 */
async function clearUserSession() {
  try {
    await ensureRedisConnected();
    await client.del("JWT");
  } catch (error) {
    console.error("Failed to clear session from Redis:", error);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Exports
// ─────────────────────────────────────────────────────────────────────────────

module.exports = {
  verifySignUp,
  checkUserSession,
  createApiSessionGuard,
  requireApiSession,
  clearUserSession,
  client,
};
