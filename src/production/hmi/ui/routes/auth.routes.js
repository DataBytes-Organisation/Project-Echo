const { verifySignUp, client } = require("../middleware");
const controller = require("../controller/auth.controller");
const emailcontroller = require('../controller/email.controller');
//const cntroller = require("../public/js/routes");
const apiClient = require('../services/apiClient');
const redis = require("redis")
require('dotenv').config();

module.exports = function (app) {
  app.use(function (req, res, next) {
    res.header(
      "Access-Control-Allow-Headers",
      "Origin, Content-Type, Accept"
    );
    next();
  });

  app.post("/api/auth/signup", verifySignUp.confirmPassword, async (req, res) => {
    
    const rolesList = [req.body.roles]
    // After signup page are completed and merged
    // Use this schema instead of the bottom one 
    let schema = {
      username : req.body.username,
      password : req.body.password,
      email : req.body.email,
      roles : rolesList,
      gender : req.body.gender,
      DoB : req.body.DoB,
      organization : req.body.organization,
      phonenumber : req.body.phonenumber,
      
      address : {"country": req.body.country, "state": req.body.state} 
    }

  
      try {
        await apiClient.post('/hmi/signup', schema)
        res.status(201).send(`<script> window.location.href = "/login"; alert("User registered successfully");</script>`);
      } catch (err) {
        // old code assumed err.response always existed here, which crashes if the API is just unreachable -
        // apiClient always gives us a consistent shape so this branch actually works now regardless of why it failed
        console.log('Signup failed: ' + err.message)
        res.status(404).send(`<script> window.location.href = "/login"; alert("Register exception error occured!");</script>`);
      }
});
    
  app.post("/api/auth/signin", async (req, res) => {
    let uname = req.body.username;
    let pw = req.body.password;

    let email = req.body.email;
      
    try {
      const data = await apiClient.post('/hmi/signin', {
        username: uname,
        email: email,
        password: pw
      });

      // Check if MFA is enabled
      if (data.mfa_phone_enabled) {
        res.status(200).send(
          `<script>
            window.location.href = "/verify-otp?user_id=${data.user_id}";
          </script>`
        );
        return;
      }

      // Normal login flow
      await client.set("JWT", data.tkn, (err, res)=> {
        if (err) {
          console.log("Set JWT Token error: ", err)
        } else {
          console.log("Set JWT successfully: ", res)
        }
      })
      await client.set("Roles", data.roles.toString(), (err, res)=> {
        if (err) {
          console.log("Set User Roles Token error: ", err)
        } else {
          console.log("Set User roles successfully: ", res)
        }
      })
      await client.set("Users", JSON.stringify(data.user), (err, res)=> {
        if (err) {
          console.log("Set User Roles Token error: ", err)
        } else {
          console.log("Set User roles successfully: ", res)
        }
      })
      res.status(200).json({
        message: "Login Successful",
        token: data.tkn,
        userId: data.user.id,
      });
    } catch (err) {
      console.error('Sign-in failed:', err.message);

      // login.html checks response.ok then does response.json() either way - it needs
      // real JSON with the right status code, not the HTML script-alert we were sending
      if (err.status === 401 || err.status === 404) {
        return res.status(401).json({
          message: 'Invalid username, email, or password.',
        });
      }

      if (err.isNetworkError) {
        return res.status(502).json({
          message: 'The sign-in service is currently unavailable.',
        });
      }

      return res.status(500).json({
        message: 'Unable to sign in. Please try again.',
      });
    }
  });



  app.post("/api/2fa/verify", async (req, res) => {
    let otp = req.body.otp;
    let user_id = req.body.user_id;
    // let email = req.body.email;
      
    try {
      const data = await apiClient.post('/2fa/verify', {
        user_id: user_id,
        otp: otp
      });

      await client.set("JWT", data.tkn, (err, res)=> {
        if (err) {
          console.log("Set JWT Token error: ", err)
        } else {
          console.log("Set JWT successfully: ", res)
        }
      })
      await client.set("Roles", data.roles.toString(), (err, res)=> {
        if (err) {
          console.log("Set User Roles Token error: ", err)
        } else {
          console.log("Set User roles successfully: ", res)
        }
      })
      await client.set("Users", JSON.stringify(data.user), (err, res)=> {
        if (err) {
          console.log("Set User Roles Token error: ", err)
        } else {
          console.log("Set User roles successfully: ", res)
        }
      })
      res.status(200).send(
      `<script>
        alert("Login Successfully");
        window.location.href = "/welcome"
      </script>`);
    } catch (err) {
      console.log('2FA verify exception error: ' + err.message);
      res.status(400).send(`<script> window.location.href = "/verify-otp?user_id=${req.body.user_id}"; alert("Failed! Invalid OTP, Please try again !");</script>`);
    }
  });

  app.post("/api/auth/forgot", async (req, res) => {
    // let account = req.body.account;
    // console.log(account)
    
    // try {
    //   const axiosResponse = await axios.post('http://ts-api-cont:9000/hmi/forgot-password', {
    //     user: account
    //   });
      
    //   if (axiosResponse.status === 201) {
    //     console.log('Status Code: ' + axiosResponse.status + ' ' + axiosResponse.statusText)
    //     console.log("Server's response: ", axiosResponse.data);

    //     enquiry = `Your new password is ${axiosResponse.data.password}`
        
    //     await emailcontroller.send_enquiry(axiosResponse.data.email, 'Recovery Password', enquiry)

    //     res.status(201).send(
    //     `<script> 
    //       alert("Password has been changed. Check your email!");
    //       window.location.href = "/login"
    //     </script>`);
          
        
    //   } else {
    //     console.log("Error response: ", axiosResponse.data);
    //     res.status(404).send('<script> window.location.href = "/login"; alert("Failed! Account not found!");</script>');
    //   }
    // } catch (err) {
    //   console.log('Exception error: ' + err)
    //   res.send(`<script> window.location.href = "/login"; alert("Exception Error: ${err}!");</script>`);
    // }  

    let account = req.body.account;
    console.log(account)
    
    try {
      const data = await apiClient.post('/hmi/forgot-password', {
        user: account
      });

      console.log("Server's response: ", data);

      enquiry = `Your new OTP is ${data.otp} and click here to reset password :- ${process.env.CLIENT_URL}/forgotPassword`

      await emailcontroller.send_enquiry(data.email, 'Recovery Password', enquiry)

      res.status(201).send(
      `<script>
        alert("Password has been changed. Check your email!");
        window.location.href = "/login"
      </script>`);
    } catch (err) {
      console.log('Forgot-password exception error: ' + err.message)
      res.status(404).send('<script> window.location.href = "/login"; alert("Failed! Account not found!");</script>');
    }
  });

  app.post("/api/auth/reset_password", async (req, res) => {
    let uname = req.body.username;
    let pw = req.body.password;
    // let _otp_ = req.body.otp;

    try {
      const data = await apiClient.post('/hmi/reset-password', {
        username: uname,
        password: pw
        // otp : _otp_
      });

      console.log("Reset response: ", data);

      res.status(201).send(
      `<script>
        alert("Login Successfully");
        window.location.href = "/welcome"
      </script>`);
    } catch (err) {
      console.log('Reset-password exception error: ' + err.message)
      res.status(404).send('<script> window.location.href = "/login"; alert("Failed! Account not found!");</script>');
    }
  });


  app.post("/api/auth/signout", controller.signout);

  app.post("/api/auth/guestsignup", controller.guestsignup);

  // app.delete("/api/auth/delete-account", controller.deleteaccount);

  // app.post("/api/auth/guestsignin", controller.guestsignin);
};
