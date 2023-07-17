const express = require('express');
const app = express();
const path = require('path');

const port = 8080;

// serve static files from the public directory
app.use(express.static(path.join(__dirname, 'public')));

//cors access to enable sending emails from different hosts
const cors = require("cors");
app.use(cors())

//bodyParser to make sure post form data is read
const bodyParser = require("express");
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }))

//const serveIndex = require('serve-index'); 
//app.use('/images/bio', serveIndex(express.static(path.join(__dirname, '/images/bio'))));

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

  let mailOptions = {
    from: email,
    to: 'echodatabytes@gmail.com',
    subject: 'Query from ' + email,
    text: query
  };
  transporter.sendMail(mailOptions, function(error, info){
    if (error) {
      console.log(error);
    } else {
      console.log('Email sent: ' + info.response);
      return res.redirect("/")
    }
  });

})

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, 'public/index.html'))
})

// start the server
// test

app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});