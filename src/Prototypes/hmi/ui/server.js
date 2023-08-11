const express = require('express');
const app = express();
const path = require('path');
const fs = require('fs');
const cookieSession = require("cookie-session");
const dbConfig = require("./config/db.config");



//Add mongoDB module inside config folder
const db = require("./model");
const Role = db.role;

//Establish Mongo Client connection to mongoDB
db.mongoose
  .connect(`mongodb://${dbConfig.USERNAME}:${dbConfig.PASSWORD}@${dbConfig.HOST}:${dbConfig.PORT}/${dbConfig.DB}`, {
    useNewUrlParser: true,
    useUnifiedTopology: true
  })
  .then(() => {
    console.log("Successfully connect to MongoDB.");
    initial();
  })
  .catch(err => {
    console.error("Connection error", err);
    process.exit();
  });


//Initalize the data if no user role existed
function initial() {
  Role.estimatedDocumentCount((err, count) => {
    if (!err && count === 0) {
      new Role({
        name: "user"
      }).save(err => {
        if (err) {
          console.log("error", err);
        }

        console.log("added 'user' to roles collection");
      });
      new Role({
        name: "admin"
      }).save(err => {
        if (err) {
          console.log("error", err);
        }

        console.log("added 'admin' to roles collection");
      });
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


app.get("/requests", (req,res) => {
  res.sendFile(path.join(__dirname, 'public/requests.html'))
})

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, 'public/index.html'))
})

// routes
require('./routes/auth.routes')(app);
require('./routes/user.routes')(app);

// start the server
app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});