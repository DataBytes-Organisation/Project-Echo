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
        // Use environment variable for API URL or fallback to localhost
        const apiUrl = process.env.API_URL || 'http://localhost:9000';
        const axiosResponse = await axios.post(`${apiUrl}/hmi/signup`,schema)
      
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
      // Try to connect to external API first
      const apiUrl = process.env.API_URL || 'http://localhost:9000';
      let useExternalAPI = true;
      
      // Check if external API is available
      try {
        await axios.get(`${apiUrl}/`, { timeout: 1000 });
      } catch (apiCheckError) {
        console.log('External API not available, using local authentication');
        useExternalAPI = false;
      }

      if (useExternalAPI) {
        // Use external API
        const axiosResponse = await axios.post(`${apiUrl}/hmi/signin`,{
          username: uname,
          email: email,
          password: pw
        });
        
        if (axiosResponse.status === 200) {
          // Check if MFA is enabled
          if (axiosResponse.data.mfa_phone_enabled) {
            res.status(200).send(
              `<script>
                window.location.href = "/verify-otp?user_id=${axiosResponse.data.user_id}";
              </script>`
            );
            return;
          }

          // Normal login flow
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
          
          res.status(200).json({
            message: "Login Successful",
            token: axiosResponse.data.tkn,
            userId: axiosResponse.data.user.id,
          });
                  
        } else {
          console.log("Login response: ", axiosResponse.data);
          res.status(400).json({ message: "Failed! Invalid credentials!" });
        }
      } else {
        // Local authentication fallback
        const bcrypt = require('bcryptjs');
        const jwt = require('jsonwebtoken');
        const { MongoClient } = require('mongodb');
        
        // Connect to MongoDB
        const mongoUri = process.env.MONGODB_URI || "mongodb://root:root_password@ts-mongodb-cont:27017";
        const mongoClient = new MongoClient(mongoUri, {
          serverSelectionTimeoutMS: 5000,
          connectTimeoutMS: 10000
        });
        
        try {
          await mongoClient.connect();
          const db = mongoClient.db('UserSample'); // Use UserSample database
          const usersCollection = db.collection('users');
          
          // Find user by username or email
          const query = uname ? { username: uname } : { email: email };
          const user = await usersCollection.findOne(query);
          
          if (!user) {
            return res.status(401).json({ message: "Invalid credentials" });
          }
          
          // Verify password
          const isValidPassword = await bcrypt.compare(pw, user.password);
          if (!isValidPassword) {
            return res.status(401).json({ message: "Invalid credentials" });
          }
          
          // Generate JWT token
          const token = jwt.sign(
            { id: user._id, username: user.username, email: user.email },
            process.env.JWT_SECRET || 'your-secret-key',
            { expiresIn: '24h' }
          );
          
          // Store in Redis
          await client.set("JWT", token);
          await client.set("Users", JSON.stringify(user));
          
          res.status(200).json({
            message: "Login Successful",
            token: token,
            userId: user._id.toString(),
          });
          
        } finally {
          await mongoClient.close();
        }
      }
    } catch (err) {
      console.log('Login exception error: ' + err);
      res.status(500).json({ message: "Login error: " + err.message });
    }  
  });



  app.post("/api/2fa/verify", async (req, res) => {
    let otp = req.body.otp;
    let user_id = req.body.user_id;
    // let email = req.body.email;
      
    try {
      // Use environment variable for API URL or fallback to localhost
      const apiUrl = process.env.API_URL || 'http://localhost:9000';
      const axiosResponse = await axios.post(`${apiUrl}/2fa/verify`,{
        user_id: user_id,
        otp: otp
      });
      
      if (axiosResponse.status === 200) {
        // Check if MFA is enabled

        // Normal login flow
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
        res.status(400).send(`<script> window.location.href = "/verify-otp?user_id=${axiosResponse.data.user_id}"; alert("Failed! Invalid OTP, Please try again !");</script>`);
      }
    } catch (err) {
        res.status(400).send(`<script> window.location.href = "/verify-otp?user_id=${req.body.user_id}"; alert("Failed! Invalid OTP, Please try again !");</script>`);
      console.log('Login exception error: ' + err);
      // res.send(`<script> window.location.href = "/login"; alert("Login exception Error: ${err}!");</script>`);
    }  
  });

  app.post("/api/auth/forgot", async (req, res) => {
    // let account = req.body.account;
    // console.log(account)
    
    // try {
    //   const axiosResponse = await axios.post('http://api-service:9000/hmi/forgot-password', {
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
      // Use environment variable for API URL or fallback to localhost
      const apiUrl = process.env.API_URL || 'http://localhost:9000';
      const axiosResponse = await axios.post(`${apiUrl}/hmi/forgot-password`, {
        user: account
      });
      
      if (axiosResponse.status === 201) {
        console.log('Status Code: ' + axiosResponse.status + ' ' + axiosResponse.statusText)
        console.log("Server's response: ", axiosResponse.data);

        enquiry = `Your new OTP is ${axiosResponse.data.otp} and click here to reset password :- http://localhost:8080/forgotPassword`
        
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

  app.post("/api/auth/reset_password", async (req, res) => {
    let uname = req.body.username;
    let pw = req.body.password;
    // let _otp_ = req.body.otp;

    try {
      // Use environment variable for API URL or fallback to localhost
      const apiUrl = process.env.API_URL || 'http://localhost:9000';
      const axiosResponse = await axios.post(`${apiUrl}/hmi/reset-password`,{
        username: uname,
        password: pw
        // otp : _otp_
      });
    
      if (axiosResponse.status === 201) {
        console.log('Status Code: ' + axiosResponse.status + ' ' + axiosResponse.statusText)
        console.log("Reset response: ", axiosResponse.data);

        res.status(201).send(
        `<script> 
          alert("Login Successfully");
          window.location.href = "/welcome"
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