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
  clearUserSession,
  client,
};
