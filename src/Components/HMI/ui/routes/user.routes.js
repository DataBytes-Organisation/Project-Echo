const express = require('express');
const router = express.Router();
const { client, authJwt } = require("../middleware");
const controller = require("../controller/user.controller");

// Middleware to set headers
router.use((req, res, next) => {
  res.header(
    "Access-Control-Allow-Headers",
    "Origin, Content-Type, Accept"
  );
  next();
});

// Define routes
router.get("/api/test/all", controller.allAccess);

router.get("/api/test/user", controller.userBoard);

router.get("/api/test/mod", controller.moderatorBoard);

router.get("/api/test/admin", controller.adminBoard);

router.get("/test", controller.publicHMI);

router.get(`/user_profile`, async (req, res) => {
  try {
    let user = await client.get('Users', (err, storedUser) => {
      if (err) {
        return `Error retrieving user role from Redis: ${err}`;
      } else {
        return storedUser;
      }
    });
    res.send(user);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Export the router
module.exports = router;
