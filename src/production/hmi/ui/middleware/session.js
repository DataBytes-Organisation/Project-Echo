"use strict";

const crypto = require("node:crypto");

function authenticationFailed(req, res) {
  if (req.path.startsWith("/api/")) {
    return res.status(401).json({ error: "Authentication required." });
  }
  return res.redirect("/login");
}

function tokensMatch(requestToken, storedToken) {
  if (typeof requestToken !== "string" || typeof storedToken !== "string") return false;

  const requestBuffer = Buffer.from(requestToken);
  const storedBuffer = Buffer.from(storedToken);
  return requestBuffer.length === storedBuffer.length &&
    crypto.timingSafeEqual(requestBuffer, storedBuffer);
}

function createCheckUserSession(redisClient, isPublicRoute = () => false) {
  return async function checkUserSession(req, res, next) {
    if (isPublicRoute(req.path)) return next();

    try {
      if (!redisClient.isOpen) await redisClient.connect();
      const storedToken = await redisClient.get("JWT");
      if (!tokensMatch(req.session?.token, storedToken)) {
        return authenticationFailed(req, res);
      }
      return next();
    } catch (error) {
      console.error("Session check failed.");
      return authenticationFailed(req, res);
    }
  };
}

module.exports = { createCheckUserSession };
