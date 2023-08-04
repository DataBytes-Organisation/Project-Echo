const express = require('express');
const app = express();
const path = require('path');
const fs = require('fs');
const cookieSession = require("cookie-session");
const dbConfig = require("./config/db.config");
const jwt = require("jsonwebtoken");
const { authJwt } = require("./middleware");
const controller = require("./controller/auth.controller");
const crypto = require("crypto")
var bcrypt = require("bcryptjs");

//Add mongoDB module inside config folder
const db = require("./model");
const Role = db.role;
const User = db.user;
const Guest = db.guest;
//Establish Mongo Client connection to mongoDB
db.mongoose
  .connect(`mongodb://${dbConfig.USERNAME}:${dbConfig.PASSWORD}@${dbConfig.HOST}/${dbConfig.DB}?authSource=admin`, {
    useNewUrlParser: true,
    useUnifiedTopology: true
  })
  .then(() => {
    console.log("Successfully connect to MongoDB.");
    initial();
    initUsers();
    initGuests();
  })
  .catch(err => {
    console.log("ConnString: ", `mongodb://${dbConfig.USERNAME}:${dbConfig.PASSWORD}@${dbConfig.HOST}/${dbConfig.DB}?authSource=admin`)
    console.error("Connection error", err);
    // process.exit();
  });


//Initalize the data if no user role existed
function initial() {
  Role.estimatedDocumentCount((err, count) => {
    if (!err && count === 0) {
      // new Role({
      //   name: "user"
      // }).save(err => {
      //   if (err) {
      //     console.log("error", err);
      //   }

      //   console.log("added 'user' to roles collection");
      // });
      // new Role({
      //   name: "admin"
      // }).save(err => {
      //   if (err) {
      //     console.log("error", err);
      //   }

      //   console.log("added 'admin' to roles collection");
      // });
      const roleData = require(path.join(__dirname, "user-sample/role-seed.json"));

      Role.insertMany(roleData);
    }
  });
}

//Add sample Users if none exists
function initUsers(){
  User.estimatedDocumentCount((err, count) => {
    if (!err && count === 0) {
      const userData = require(path.join(__dirname, "user-sample/user-seed.json"));
      User.insertMany(userData);
      
    }
  });
}

//Add sample Guest users if none exists
function initGuests(){
  Guest.estimatedDocumentCount((err, count) => {
    if (!err && count === 0) {
      //Different from Roles and Users, 
      // another approach is to manually seed mongoDB document
      // Only feasible if there are only 1-2 sample documents
      const newGuest1 = new Guest({
        userId: "HMITest1",
        username: "guest_tester1_0987654321",
        email: "guest@echo.com",
        password: bcrypt.hashSync("guest_password", 8),
        roles: [
          {
            "_id":"64be1d0f05225843178d91d7"
          }
        ],
        expiresAt: new Date(Date.now() + 1800000) // Set the expiration duration for 30 mins = 1800 s = 1800000 ms from now
      });
      
      // Save the new guest document to the collection
      newGuest1.save((err, doc) => {
        if (err) {
          console.error(err);
        } else {
          console.log('Guest document inserted successfully:', doc);
        }
      });

      const newGuest2 = new Guest({
        userId: "HMITest2",
        username: "guest_tester2_1234567890",
        email: "guest@hmi.com",
        password: bcrypt.hashSync("guest_password", 8),
        roles: [
          {
            "_id":"64be1d0f05225843178d91d7"
          }
        ],
        expiresAt: new Date(Date.now() + 300000) // Set the expiration duration for 5 mins = 300 s = 300000 ms from now
      });
      
      // Save the new guest document to the collection
      newGuest2.save((err, doc) => {
        if (err) {
          console.error(err);
        } else {
          console.log('Guest document inserted successfully:', doc);
        }
      });
    }
  });
}

//Background Process to automatically delete Guest role after exceeding expiration
setInterval(() => {
  const now = new Date();
  console.log("Background monitor at ", now.toString())
  Guest.deleteMany({ expiresAt: { $lte: now } }, (err) => {
    if (err) {
      console.error('Error deleting expired documents:', err);
    } else {
      console.log('Expired documents deleted successfully.');
    }
  });
}, 360000); // Run every 6 mins = 360 s = 360000 ms (adjust as needed)

const port = 8080;

// serve static files from the public directory
app.use(express.static(path.join(__dirname, 'public')));

//cors access to enable sending emails from different hosts
const cors = require("cors");

var corsOptions = {
  origin: "http://localhost:8081"
};

app.use(cors(corsOptions))

//bodyParser to make sure post form data is read
const bodyParser = require("express");
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }))

//const serveIndex = require('serve-index'); 
//app.use('/images/bio', serveIndex(express.static(path.join(__dirname, '/images/bio'))));

app.use(
  cookieSession({
    name: "echo-session",
    keys: ["COOKIE_SECRET"], // should use as secret environment variable
    httpOnly: true
  })
);

const nodemailer = require('nodemailer');
var transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: 'echodatabytes@gmail.com',
    pass: 'ltzoycrrkpeipngi'
  }
});

app.post("/send_email", (req,res) => {
  const {email, query} = req.body;
  let html_text = '<div>';
  html_text += '<h2>A new query has been received for Project Echo HMI</h2>'
  html_text += '<img src="cid:logo@echo.hmi" style="height: 150px; width: 150px; display: flex; margin: auto;"/>'
  html_text += '<p>Sender: \t ' + email + '</p>';
  html_text += '<p>Query: \t ' + query + '</p>';
  html_text += '<hr>';
  html_text += '<p>Yes, this mailbox is active. So please feel free to reply to this email if you have other queries.</p>'
  
  html_text += '</div>';
  let mailOptions = {
    from: email,
    to: `echodatabytes@gmail.com, ${email}, databytes@deakin.edu.au`,
    subject: 'New query received!',
    text: query,
    html: html_text,
    attachments: [{   // stream as an attachment
      filename: 'image.png',
      content: fs.createReadStream(path.join(__dirname, 'public/images/tabIcons/logo.png')),
      cid: 'logo@echo.hmi' //same cid value as in the html
  }]
  }
  transporter.sendMail(mailOptions, function(error, info){
    if (error) {
      console.log(error);
    } else {
      console.log('Email sent: ' + info.response);
      return res.redirect("/")
    }
  });

})

var chars = "0123456789abcdefghijklmnopqrstuvwxyz!@#$%^&*()ABCDEFGHIJKLMNOPQRSTUVWXYZ";

function genPass(length){
  let password = "";
  for (var i = 0; i <= parseInt(length); i++) {
    var randomNumber = Math.floor(Math.random() * chars.length);
    password += chars.substring(randomNumber, randomNumber +1);
   }
  return password;
}

app.post("/request_access", async (req,res) => {
  console.log("email: ", req.body.email);
  const {email} = req.body;
  //Generate Guest credentials + timestamp
  let username = 'guest_' + email.split('@')[0] + "_" + crypto.getRandomValues(new Uint32Array(1));
  console.log("username: ", username)
  let password =  genPass(12);
  let timestamp = new Date(Date.now() + 1800000) //Set time to live of 1800000 ms = 1800 s = 30 mins
  let request = {
    "username": username,
    "email": req.body.email,
    "password": password,
    "timestamp": timestamp
  }
  try {
    //Sending that to Guest signup
    const response = await controller.guestsignup(request);
    
    setTimeout(()=>{
      console.log("response is back! ", response);
      //Send email to user when success
      if (response && response.status === 'success') {
        let html_text = '<div>';
        html_text += '<h2>Echo HMI Temporary Access Requested!</h2>'
        html_text += '<img src="cid:logo@echo.hmi" style="height: 150px; width: 150px; display: flex; margin: auto;"/>'
        html_text += '<p>Dear \t <strong>' + req.body.email + '</strong></p>';
        html_text += '<hr>';
        html_text += '<p>Thank you for your patience, here is your login credential </p>'
        html_text += '<p><strong>Username:</strong> \t ' + username + '</p>'
        html_text += '<p><strong>Password:</strong> \t ' + password + '</p>'
        html_text += '<br><p>Please take in mind that this account will only be valid until '+ timestamp.toString() + ' (Subject to change based on development)</p>'
        html_text += '</div>';
        let mailOptions = {
          from: email,
          to: `echodatabytes@gmail.com, ${email}`,
          subject: 'Guest User Access Granted!',
          html: html_text,
          attachments: [{   // stream as an attachment
            filename: 'image.png',
            content: fs.createReadStream(path.join(__dirname, 'public/images/tabIcons/logo.png')),
            cid: 'logo@echo.hmi' //same cid value as in the html
        }]
        }
        transporter.sendMail(mailOptions, function(error, info){
          if (error) {
            console.log(error);
          } else {
            console.log('Email sent: ' + info.response);
            return res.redirect("/login")
          }
        });
      } else {
        console.log("Something happened for Guest Access Granting: ", response);
        let error_box = document.getElementById("request-access-email-error");
        error_box.innerHTML = `Exception error occured: ${response.message}`;
        error_box.style.display = "block"
        setTimeout(()=> {
          error_box.innerHTML = '';
          error_box.style.display = "none";
        },3000)
      }

    },200)
    
  } catch (error) {
    res.status(500).send({ message: 'An error occurred while sending the request access: ' + error });
  }
  
 


  

})

// routes
require('./routes/auth.routes')(app);
require('./routes/user.routes')(app);

app.get("/", (req, res) => {
  if (authJwt.verifyToken && authJwt.isUser) {
    console.log("This is user session!")
    res.sendFile(path.join(__dirname, 'public/index.html'))
  }
  else {
    console.log("This is not user sessions!")
    res.json("No available session, you have not logged in yet")
  }
  
})



app.get("/login", (req,res) => {
  res.sendFile(path.join(__dirname, 'public/login.html'));
})


app.get("*", (req,res) => {
  let token = req.session.token;
  console.log("Current token: ", token)
  if (!token) {
    console.log("Current user session unavailable")
    res.sendFile(path.join(__dirname, 'public/login.html'));
  } else {
    console.log("redirect to homepage")
    return res.sendFile(path.join(__dirname, 'public/index.html'))
  }

})

// start the server
app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});