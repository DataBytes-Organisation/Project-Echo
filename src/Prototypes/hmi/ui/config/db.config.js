module.exports = {
  USERNAME: process.env.MONGODB_USER || "<env-required>",
  PASSWORD: process.env.MONGODB_PASS || "<env-required>",
  HOST: process.env.MONGODB_HOST || "localhost",
  PORT: parseInt(process.env.MONGODB_PORT || "27017", 10),
  DB: process.env.MONGODB_DB || "UserSample"
};

