const assert = require("node:assert/strict");
const test = require("node:test");

test("start() still listens when the donation database is unreachable", async () => {
  process.env.PORT = "0";
  const dbPath = require.resolve("../../server/services/database");
  const redisPath = require.resolve("../../server/services/redis");
  const dbModule = require(dbPath);
  const redisModule = require(redisPath);
  const originalConnect = dbModule.connectDatabases;
  const originalCreate = redisModule.createRedisClient;
  dbModule.connectDatabases = async () => {
    throw new Error("stubbed donation DB outage");
  };
  redisModule.createRedisClient = () => ({
    connect: async () => {},
    quit: async () => {},
  });
  try {
    const { start } = require("../../server");
    const { server, close } = await start();
    try {
      assert.equal(server.listening, true);
    } finally {
      await close();
    }
  } finally {
    dbModule.connectDatabases = originalConnect;
    redisModule.createRedisClient = originalCreate;
    delete process.env.PORT;
  }
});
