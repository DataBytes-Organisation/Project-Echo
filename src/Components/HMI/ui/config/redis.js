// config/redis.js
const { createClient } = require('@redis/client');

const redisClient = createClient({
  url: process.env.REDIS_URL || 'redis://localhost:6379'
});

redisClient.on('error', (err) => {
  console.error('Redis Client Error', err);
});

async function connectRedis() {
  if (!redisClient.isOpen) {
    await redisClient.connect();
    console.log('✅ Redis connected');
  }
}

module.exports = {
  redisClient,
  connectRedis
};
