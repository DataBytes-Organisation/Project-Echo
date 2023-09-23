const { verifySignUp } = require("../middleware");
const controller = require("../controller/auth.controller");

module.exports = function(app) {
  app.use(function(req, res, next) {
    res.header(
      "Access-Control-Allow-Headers",
      "Origin, Content-Type, Accept"
    );
    next();
  });

  app.post(
    "/api/auth/signup",
    [
      verifySignUp.checkDuplicateUsernameOrEmail,
      verifySignUp.checkRolesExisted
    ],
    controller.signup
  );

  app.post("/api/auth/signin", controller.signin);

  app.post("/api/auth/signout", controller.signout);

  app.post("/api/auth/verifyForget", controller.verifyForget);

  app.get("/forgetPass",controller.loadForget);

  app.post("/forgetPass",controller.verifyForget);

  app.get("/newPass", controller.loadNewPass);

  app.post("/newPass", controller.makeNewPass);
};