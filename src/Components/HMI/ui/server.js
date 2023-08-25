const express = require('express');
const app = express();
const path = require('path');
const fs = require('fs');
const cookieSession = require("cookie-session");
const dbConfig = require("./config/db.config");
const jwt = require("jsonwebtoken");
const { authJwt } = require("./middleware");



//Add mongoDB module inside config folder
const db = require("./model");
const Role = db.role;
const User = db.user;
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

const port = 8080;

// serve static files from the public directory
app.use(express.static(path.join(__dirname, 'public')));

//cors access to enable sending emails from different hosts
const cors = require("cors");

var corsOptions = {
  origin: "http://localhost:8080"
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

app.get("/forgetPass", (req,res) => {
  res.sendFile(path.join(__dirname, 'public/forgetPass.html'));
})


app.get("*", (req,res) => {
  let token = req.session.token;
  console.log("Current token: ", token)
  if (!token) {
    console.log("Current user session unavailable")
    res.sendFile(path.join(__dirname, 'public/login.html'));
    res.sendFile(path.join(__dirname, 'public/forgetPass.html'));
  } else {
    console.log("redirect to homepage")
    return res.sendFile(path.join(__dirname, 'public/index.html'))
  }

})

// start the server
app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});