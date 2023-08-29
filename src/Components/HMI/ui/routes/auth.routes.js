const { verifySignUp } = require("../middleware");
const controller = require("../controller/auth.controller");
//const cntroller = require("../public/js/routes");
const axios = require('axios');

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
    
    app.post("/api/auth/signin", async (req, res) => {
      // let uname = req.body.username;
      // let re = new RegExp("^guest\_.*\_[0-9]{8,10}$");
      
      // //Check user login based on Regex Pattern:
      // if (uname.match(re) === null) {
      //   console.log("username is not from Guest, proceed to User Signin")
      //   controller.signin(req, res)
      //   // const response = await cntroller.signin(req, res)
      //   // res.json(response)
      // } else {
      //   console.log("username is from Guest, proceed to Guest Signin")
      //   controller.guestsignin(req, res)
      // }


      // controller.signin
      //cntroller.signin

      // let username = req.body.username;
      // let password = req.body.pw;
      // // const response = await axios.post('http://localhost:9000/hmi/signin', {
      // //   username: username,
      // //   password: password
      // // })
      // // console.log(response)
      // axios.post('http://localhost:9000/hmi/signin', {
      //     username: username,
      //     password: password
      //   }).then(res => {
      //     res.status(200).redirect("/welcome")
      //   })
      //     .catch(err => console.log(err))

      
      let username = req.body.username;
      let password = req.body.password;
      try {
        const axiosResponse = await axios.post('http://localhost:9000/hmi/signin', {
            username: username,
            password: password
        });
        console.log("testing a")
        if (axiosResponse.status === 200) {
            res.status(200).redirect("/welcome")
        } else {
            res.status(500).json({ message: "Unexpected response from the external service." });
        }
    } catch (err) {
      console.log("testing b")
      console.log(err)
      res.status(500).json({ message: "Error occurred while making the external request." });
    }
      
      
  });

  app.post("/api/auth/signout", controller.signout);

  app.post("/api/auth/guestsignup", controller.guestsignup);

  app.post("/api/auth/guestsignin", controller.guestsignin);
};