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
  console.log("Get user session: ", req.path)
  if (req.path == '/welcome' || req.path == '/' || req.path == '/map' | req.path.includes("admin") | req.path == null | req.path == undefined) {

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
  }
  return next()
}

module.exports = {
  verifySignUp,
  checkUserSession,
  client
};