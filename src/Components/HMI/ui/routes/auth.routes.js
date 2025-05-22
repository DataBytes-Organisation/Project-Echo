const { verifySignUp, client } = require("../middleware");
const controller = require("../controller/auth.controller");
const emailcontroller = require('../controller/email.controller');
const axios = require('axios');
const redis = require("redis");

module.exports = function (app) {
  app.use(function (req, res, next) {
    res.header(
      "Access-Control-Allow-Headers",
      "Origin, Content-Type, Accept"
    );
    next();
  });

  app.post("/api/auth/signup", verifySignUp.confirmPassword, async (req, res) => {
    const rolesList = [req.body.roles];
    let schema = {
      username: req.body.username,
      password: req.body.password,
      email: req.body.email,
      roles: rolesList,
      gender: req.body.gender,
      DoB: req.body.DoB,
      organization: req.body.organization,
      phonenumber: req.body.phonenumber,
      address: { country: req.body.country, state: req.body.state }
    };

    try {
      const axiosResponse = await axios.post('http://localhost:9000/hmi/signup', schema);

      if (axiosResponse.status === 201) {
        console.log('Signup OK:', axiosResponse.status);
        res.status(201).send(`<script> alert("User registered successfully"); window.location.href = "/login"; </script>`);
      } else {
        res.status(400).send(`<script> alert("Ooops! Something went wrong"); window.location.href = "/login"; </script>`);
      }
    } catch (err) {
      console.log('Signup exception:', err.response?.status, err.response?.data);
      res.status(404).send(`<script> alert("Register exception occurred!"); window.location.href = "/login"; </script>`);
    }
  });

  app.post("/api/auth/signin", async (req, res) => {
    const uname = req.body.username;
    const pw = req.body.password;

    if (!uname || !pw) {
      return res.status(400).send(`<script>alert("Username or password missing."); window.location.href="/login";</script>`);
    }

    console.log("POSTING TO BACKEND WITH:", { username: uname, password: pw });

    try {
      const axiosResponse = await axios.post('http://localhost:9000/hmi/signin', {
        username: uname,
        password: pw,
        email: uname
      });

      if (axiosResponse.status === 200) {
        const data = axiosResponse.data;
        console.log("Login response:", data);

        await client.set("JWT", data.tkn);
        await client.set("Roles", data.roles.toString());
        await client.set("Users", JSON.stringify(data.user));

        res.status(200).json({
          token: data.tkn,
          userId: data.user.id
        });
      } else {
        res.status(400).send(`<script> alert("Failed! Invalid credentials!"); window.location.href = "/login"; </script>`);
      }
    } catch (err) {
      console.log('Login exception error:', err.response?.status, err.response?.data);
      res.send(`<script> alert("Login exception occurred!"); window.location.href = "/login"; </script>`);
    }
  });

  app.post("/api/auth/forgot", async (req, res) => {
    let account = req.body.account;
    console.log("Forgot-password input:", account);

    try {
      const axiosResponse = await axios.post('http://localhost:9000/hmi/forgot-password', {
        user: account
      });

      if (axiosResponse.status === 201) {
        const enquiry = `Your new OTP is ${axiosResponse.data.otp} and click here to reset password: http://localhost:8080/forgotPassword`;
        await emailcontroller.send_enquiry(axiosResponse.data.email, 'Recovery Password', enquiry);

        res.status(201).send(`<script>alert("Password has been changed. Check your email!"); window.location.href = "/login";</script>`);
      } else {
        res.status(404).send(`<script>alert("Failed! Account not found!"); window.location.href = "/login";</script>`);
      }
    } catch (err) {
      console.log('Forgot exception:', err.response?.status, err.response?.data);
      res.send(`<script>alert("Exception Error occurred!"); window.location.href = "/login";</script>`);
    }
  });

  app.post("/api/auth/reset_password", async (req, res) => {
    let uname = req.body.username;
    let pw = req.body.password;

    try {
      const axiosResponse = await axios.post('http://localhost:9000/hmi/reset-password', {
        username: uname,
        password: pw
      });

      if (axiosResponse.status === 201) {
        res.status(201).send(`<script>alert("Password reset successfully."); window.location.href = "/welcome";</script>`);
      } else {
        res.status(404).send(`<script>alert("Failed! Account not found!"); window.location.href = "/login";</script>`);
      }
    } catch (err) {
      console.log('Reset password exception:', err.response?.status, err.response?.data);
      res.send(`<script>alert("Exception Error occurred!"); window.location.href = "/login";</script>`);
    }
  });

  app.post("/api/auth/signout", controller.signout);
  app.post("/api/auth/guestsignup", controller.guestsignup);
};