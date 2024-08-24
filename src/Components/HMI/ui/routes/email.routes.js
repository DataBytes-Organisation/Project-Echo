// routes/email.routes.js
const express = require('express');
const router = express.Router();
const nodemailer = require('nodemailer');
const fs = require('fs');
const path = require('path');
const validation = require('deep-email-validator');

const transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: process.env.USER_EMAIL,
    pass: process.env.PASSWORD,
  },
});

async function testEmail(input) {
  let res = await validation.validate(input);
  return { result: res.valid, response: res.validators };
}

router.post("/send_email", async (req, res) => {
  const { email, query } = req.body;
  const validationResult = await testEmail(email);
  if (validationResult.result) {
    let html_text = `<div>
      <h2>A new query has been received for Project Echo HMI</h2>
      <img src="cid:logo@echo.hmi" style="height: 150px; width: 150px; display: flex; margin: auto;"/>
      <p>Sender: ${email}</p>
      <p>Query: ${query}</p>
      <hr>
      <p>Yes, this mailbox is active. So please feel free to reply to this email if you have other queries.</p>
    </div>`;
    
    let mailOptions = {
      from: email,
      to: `${process.env.USER_EMAIL}, ${email}`,
      subject: 'New query received!',
      text: query,
      html: html_text,
      attachments: [{   
        filename: 'image.png',
        content: fs.createReadStream(path.join(__dirname, '../public/images/tabIcons/logo.png')),
        cid: 'logo@echo.hmi'
      }]
    };
    
    transporter.sendMail(mailOptions, function (error, info) {
      if (error) {
        console.log(error);
      } else {
        res.send('<script> alert("User query sent! Please check your mailbox for further communication"); window.location.href = "/"; </script>');
      }
    });
  } else {
    res.status(400).send("<script> alert(`Sender's email is not valid!`)</script>");
  }
});

module.exports = router;
