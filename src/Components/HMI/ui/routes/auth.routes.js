const { verifySignUp } = require("../middleware");
const controller = require("../controller/auth.controller");

module.exports = function (app) {
  app.use(function (req, res, next) {
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
      verifySignUp.checkRolesExisted,
      verifySignUp.confirmPassword
    ],
    controller.signup
    );
    
    app.post("/api/auth/signin", (req, res) => {
      let uname = req.body.username;
      let re = new RegExp("^guest\_.*\_[0-9]{8,10}$");
      
      //Check user login based on Regex Pattern:
      if (uname.match(re) === null) {
        console.log("username is not from Guest, proceed to User Signin")
        controller.signin(req, res)
      } else {
        console.log("username is from Guest, proceed to Guest Signin")
        controller.guestsignin(req, res)
      }
      controller.signin

  });

  app.post("/api/auth/signout", controller.signout);

  app.post("/api/auth/guestsignup", controller.guestsignup);

  app.post("/api/auth/guestsignin", controller.guestsignin);
};