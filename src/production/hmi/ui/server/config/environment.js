const crypto = require("node:crypto");

function loadEnvironment(env = process.env) {
  return {
    port: Number(env.PORT || 3000),
    apiHost: env.API_HOST || "localhost",
    apiPort: Number(env.API_PORT || 9000),
    clientUrl: env.CLIENT_URL || "http://localhost:3000",
    cookieSecret: env.COOKIE_SECRET || crypto.randomBytes(32).toString("hex"),
    redisHost: env.REDIS_HOST || "localhost",
    redisPort: Number(env.REDIS_PORT || 6379),
    mongodbUri: env.MONGODB_URI || `mongodb://${env.DB_HOST || "localhost"}:27017`,
    userMongodbUri: env.USER_MONGODB_URI,
  };
}

module.exports = { loadEnvironment };
