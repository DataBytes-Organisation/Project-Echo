require("dotenv").config();

const { createApp } = require("./server/app");
const { loadEnvironment } = require("./server/config/environment");
const { connectDatabases } = require("./server/services/database");
const { createRedisClient } = require("./server/services/redis");

async function start() {
  const config = loadEnvironment();
  const redisClient = createRedisClient(config);
  const databases = await connectDatabases(config);
  await redisClient.connect();
  const app = createApp({ config, redisClient, databases });
  const server = app.listen(config.port, () => {
    console.log(`Server listening on port ${config.port}`);
  });

  const close = async () => {
    server.close();
    await Promise.allSettled([redisClient.quit(), databases.close()]);
  };
  process.once("SIGINT", close);
  process.once("SIGTERM", close);
  return { app, server, close };
}

if (require.main === module) {
  start().catch((error) => {
    console.error("HMI failed to start:", error.message);
    process.exitCode = 1;
  });
}

module.exports = { start };
