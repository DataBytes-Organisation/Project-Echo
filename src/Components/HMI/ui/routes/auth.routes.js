const { verifySignUp, client } = require("../middleware");
const controller = require("../controller/auth.controller");
const emailcontroller = require('../controller/email.controller');
//const cntroller = require("../public/js/routes");
const axios = require('axios');
const redis = require("redis")

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
        const axiosResponse = await axios.post('http://ts-api-cont:9000/hmi/signup',schema)
      
        if (axiosResponse.status === 201) {
          console.log('Status Code: ' + axiosResponse.status + ' ' + axiosResponse.statusText)
          res.status(201).send(`<script> window.location.href = "/login"; alert("User registered successfully");</script>`);
        } else {
          res.status(400).send(`<script> window.location.href = "/login"; alert("Ooops! Something went wrong");</script>`);
        }
      } catch (err) {
        console.log('Status Code: ' + err.response.status + ' ' + err.response.statusText)
        console.log(err.response.data)
        res.status(404).send(`<script> window.location.href = "/login"; alert("Register exception error occured!");</script>`);
      }
});
    
  app.post("/api/auth/signin", async (req, res) => {
    let uname = req.body.username;
    let pw = req.body.password;

    let email = req.body.email;
      
    try {
      const axiosResponse = await axios.post('http://ts-api-cont:9000/hmi/signin',{
        username: uname,
        email: email,
        password: pw
      });
      
      if (axiosResponse.status === 200) {
        console.log('Status Code: ' + axiosResponse.status + ' ' + axiosResponse.statusText)
        console.log("Login response: ", axiosResponse.data);
        
        await client.set("JWT", axiosResponse.data.tkn, (err, res)=> {
          if (err) {
            console.log("Set JWT Token error: ", err)
          } else {
            console.log("Set JWT successfully: ", res)
          }
        })
        await client.set("Roles", axiosResponse.data.roles.toString(), (err, res)=> {
          if (err) {
            console.log("Set User Roles Token error: ", err)
          } else {
            console.log("Set User roles successfully: ", res)
          }
        })
        await client.set("Users", JSON.stringify(axiosResponse.data.user), (err, res)=> {
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
          
        
      } else {
        console.log("Login response: ", axiosResponse.data);
        res.status(400).send('<script> window.location.href = "/login"; alert("Failed! Invalid credentials!");</script>');
      }
    } catch (err) {
      console.log('Login exception error: ' + err)
      res.send(`<script> window.location.href = "/login"; alert("Login exception Error: ${err}!");</script>`);
    }  
  });

  app.post("/api/auth/forgot", async (req, res) => {
    let account = req.body.account;
    console.log(account)
    
    try {
      const axiosResponse = await axios.post('http://ts-api-cont:9000/hmi/forgot-password', {
        user: account
      });
      
      if (axiosResponse.status === 201) {
        console.log('Status Code: ' + axiosResponse.status + ' ' + axiosResponse.statusText)
        console.log("Server's response: ", axiosResponse.data);

        enquiry = `Your new password is ${axiosResponse.data.password}`
        
        await emailcontroller.send_enquiry(axiosResponse.data.email, 'Recovery Password', enquiry)

        res.status(201).send(
        `<script> 
          alert("Password has been changed. Check your email!");
          window.location.href = "/login"
        </script>`);
          
        
      } else {
        console.log("Error response: ", axiosResponse.data);
        res.status(404).send('<script> window.location.href = "/login"; alert("Failed! Account not found!");</script>');
      }
    } catch (err) {
      console.log('Exception error: ' + err)
      res.send(`<script> window.location.href = "/login"; alert("Exception Error: ${err}!");</script>`);
    }  
 
});

  app.post("/api/auth/signout", controller.signout);

  app.post("/api/auth/guestsignup", controller.guestsignup);

  // app.delete("/api/auth/delete-account", controller.deleteaccount);

  // app.post("/api/auth/guestsignin", controller.guestsignin);
};