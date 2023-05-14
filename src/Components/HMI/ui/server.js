const express = require('express');
const app = express();
const path = require('path');
const fs = require('fs');

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

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, 'public/index.html'))
})

// start the server
app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});