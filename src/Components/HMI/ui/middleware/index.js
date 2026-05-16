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

// ─────────────────────────────────────────────────────────────────────────────
// Redis client
// ─────────────────────────────────────────────────────────────────────────────

const client = redis.createClient({
  socket: {
    host: "localhost",
    port: 6379,
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
// All other routes are treated as protected and require a valid JWT in Redis.
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

/**
 * Express middleware that checks for a valid JWT stored in Redis.
 *
 * - Public routes (defined above) pass through without a token check.
 * - All other routes require a JWT to be present in Redis under the key "JWT".
 * - If no token is found, or Redis errors, the request is redirected to /login.
 *
 * @param {import('express').Request}  req
 * @param {import('express').Response} res
 * @param {import('express').NextFunction} next
 */
async function checkUserSession(req, res, next) {
  console.log("Session check:", req.path);

  // Public routes skip the token check entirely
  if (_isPublicRoute(req.path)) {
    return next();
  }

  // Protected route — verify a token exists in Redis
  try {
    await ensureRedisConnected();

    const token = await client.get("JWT");

    if (!token) {
      console.log("No stored token — redirecting to login");
      return res.redirect("/login");
    }

    return next();
  } catch (error) {
    console.error("Session check failed (Redis error):", error);
    return res.redirect("/login");
  }
}

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
