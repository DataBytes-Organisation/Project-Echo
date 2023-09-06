const authJwt = require("./authJwt");
const verifySignUp = require("./verifySignup");
const redis = require('redis');
const client = redis.createClient();
module.exports = {
  authJwt,
  verifySignUp,
  client
};