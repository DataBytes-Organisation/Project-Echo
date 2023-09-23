const {client, authJwt } = require("../middleware");
const controller = require("../controller/user.controller");

module.exports = function(app) {
  app.use(function(req, res, next) {
    res.header(
      "Access-Control-Allow-Headers",
      "Origin, Content-Type, Accept"
    );
    next();
  });

  app.get("/api/test/all", controller.allAccess);

  app.get("/api/test/user", controller.userBoard);

  app.get(
    "/api/test/mod",
    controller.moderatorBoard
  );

  app.get(
    "/api/test/admin",
    controller.adminBoard
  );
  app.get(
    "/test",
    controller.publicHMI
  )
  app.get(`/user_profile`, async (req, res, next) => {
    let user = await client.get('Users', (err, storedUser) => {
      if (err) {
        return `Error retrieving user role from Redis: ${err}`
      } else {
        return storedUser
      }
    })
    
    res.send(user);
    next()
  })
};