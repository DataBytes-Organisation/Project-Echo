const verifySignUp = require("./verifySignup");
const redis = require('redis');

// ✅ Updated Redis config for local development
const redisConfig = {
  host: 'localhost',
  port: 6379,
};

// ✅ Create Redis client using localhost
const client = redis.createClient({
  socket: {
    host: redisConfig.host,
    port: redisConfig.port
  }
});

// Connect the Redis client
client.connect().catch(console.error);

// Middleware to check user session
async function checkUserSession(req, res, next) {
  console.log("Get user session: ", req.path);
  if (
    req.path === '/welcome' ||
    req.path === '/' ||
    req.path === '/map' ||
    req.path.includes("admin") ||
    req.path == null ||
    req.path == undefined
  ) {
    try {
      const token = await client.get('JWT');
      if (!token) {
        console.log("No stored token, return to login");
        return res.redirect('/login');
      }
    } catch (err) {
      console.error('Error retrieving token from Redis:', err);
      return res.redirect('/login');
    }
  }
  return next();
}

module.exports = {
  verifySignUp,
  checkUserSession,
  client
};