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

  // app.post(
  //   "/api/auth/signup",
  //   [
  //    verifySignUp.checkDuplicateUsernameOrEmail,
  //    verifySignUp.checkRolesExisted,
  //    verifySignUp.confirmPassword
  //   ],
  //    controller.signup
        
  
  //   );
  app.post("/api/auth/signup", verifySignUp.confirmPassword, async (req, res) => {
    
    
    // // After signup page are completed and merged
    // // Use this schema instead of the bottom one 
    // let schema = {
    //   uname : req.body.username,
    //   pw : req.body.password,
    //   email : req.body.email,
    //   roles : req.body.roles,
    //   gender : req.body.gender,
    //   DoB : req.body.DoB,
    //   organization : req.body.organization,
    //   phonenumber : req.body.phonenumber,
      
    //   address : {"country": req.body.country, "state": req.body.state}   
    // }

    // Testing and working 
    let schema = {
      username : req.body.username,
      password : req.body.password,
      email : req.body.email,
      roles : ['admin', 'user'],
      gender : 'M',
      DoB : '01/01/2001',
      organization : 'Deakin',
      phonenumber : '0400000000',
      
      address : {"country": 'Australia', "state": 'VIC'}   
    }
      try {
        const axiosResponse = await axios.post('http://ts-api-cont:9000/hmi/signup',schema)
      
        if (axiosResponse.status === 201) {
          console.log('Status Code: ' + axiosResponse.status + ' ' + axiosResponse.statusText)
          res.status(201).redirect('/welcome')
        } 
      } catch (err) {
        console.log('Status Code: ' + err.response.status + ' ' + err.response.statusText)
        console.log(err.response.data)
        res.status(err.response.status).redirect('/login')
      }
});
    
  app.post("/api/auth/signin", async (req, res) => {
      let uname = req.body.username;
      let pw = req.body.password;

      let email = req.body.email;
      
      let re = new RegExp("^guest\_.*\_[0-9]{8,10}$");
      
      //Check user login based on Regex Pattern:
      if (uname.match(re) === null) {
        console.log("username is not from Guest, proceed to User Signin")
        try {
          const axiosResponse = await axios.post('http://ts-api-cont:9000/hmi/signin', {
            username: uname,
            email: email,
            password: pw
          });
          
          if (axiosResponse.status === 200) {
            console.log('Status Code: ' + axiosResponse.status + ' ' + axiosResponse.statusText)
            res.status(200).redirect('/welcome')
          } 
        } catch (err) {
          console.log('Status Code: ' + err.response.status + ' ' + err.response.statusText)
          console.log(err.response.data)
          res.status(err.response.status).redirect('/login')
        }  
      } 
      else {
        console.log("username is from Guest, proceed to Guest Signin")
        controller.guestsignin(req, res)
      }
  });

  app.post("/api/auth/signout", controller.signout);

  app.post("/api/auth/guestsignup", controller.guestsignup);

  app.post("/api/auth/guestsignin", controller.guestsignin);
};