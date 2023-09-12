const authJwt = require("./authJwt");
const verifySignUp = require("./verifySignup");
const redis = require('redis');

const redisConfig = {
  host: 'echo-redis',
  port: 6379,
};
const client = redis.createClient({
  socket: {
      host: 'echo-redis',
      port: 6379
  }
});

async function checkUserSession(req, res, next) {
  if ( req.path == '/login') return next();
  let token = await client.get('JWT', (err, storedToken) => {
    if (err) {
      console.error('Error retrieving token from Redis:', err);
      return null
    } else {
      console.log('Stored Token:', storedToken);
      return storedToken
    }
  })
  if (token == null) {
    // Token is missing, redirect the user to the login page
    console.log("No stored token, return to login")
    return res.redirect('/login');
  }
  next()
}

module.exports = {
  authJwt,
  verifySignUp,
  checkUserSession,
  client
};