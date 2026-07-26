const authJwt = require("./authJwt");
const verifySignUp = require("./verifySignup");
const redis = require("redis");
require("dotenv").config();

const redisConfig = {
  host: `${process.env.REDIS_HOST}`,
  port: 6379,
};
const client = redis.createClient({
  // username: 'admin', // use your Redis user. More info https://redis.io/docs/management/security/acl/
  // password: 'root', // use your password here
  socket: {
    host: `${process.env.REDIS_HOST}`,
    port: 6379,
    // tls: true,
    // key: readFileSync('./redis_user_private.key'),
    // cert: readFileSync('./redis_user.crt'),
    // ca: [readFileSync('./redis_ca.pem')]
  },
});

module.exports = {
  authJwt,
  verifySignUp,
  client,
};
