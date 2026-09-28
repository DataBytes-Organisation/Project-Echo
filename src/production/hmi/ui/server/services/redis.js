const { client } = require("../middleware");

function createRedisClient() {
  return client;
}

module.exports = { createRedisClient };
