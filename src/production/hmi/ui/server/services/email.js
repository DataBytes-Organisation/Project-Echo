const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const nodemailer = require('nodemailer');
const validation = require('deep-email-validator');

function createEmailService({ rootDirectory, controller, transporter } = {}) {
  const mailer = transporter || nodemailer.createTransport({
    service: process.env.EMAIL_SERVICE || 'gmail',
    auth: {
      user: process.env.EMAIL_USER || 'echodatabytes@gmail.com',
      pass: process.env.EMAIL_PASSWORD,
    }
  });

  function escapeHtmlEntities(input) {
    return input.replace(/[\u00A0-\u9999<>&]/gim, function (i) {
      return '&#' + i.charCodeAt(0) + ';';
    });
  }

  async function testEmail(input) {
    let result = await validation.validate(input);
    return { result: result.valid, response: result.validators };
  }

  function logoAttachment() {
    return [{
      filename: 'image.png',
      content: fs.createReadStream(path.join(rootDirectory, 'public/images/tabIcons/logo.png')),
      cid: 'logo@echo.hmi'
    }];
  }

  function sendMail(options, callback) {
    return mailer.sendMail(options, callback);
  }

  function genPass(length) {
    const chars = '0123456789abcdefghijklmnopqrstuvwxyz!@#$%^&*()ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    let password = '';
    for (let i = 0; i <= parseInt(length); i++) {
      const randomNumber = Math.floor(Math.random() * chars.length);
      password += chars.substring(randomNumber, randomNumber + 1);
    }
    return password;
  }

  function guestSalt() {
    let salt = '';
    while (salt.length < 8) {
      salt = crypto.getRandomValues(new Uint32Array(1)).toString();
    }
    return salt;
  }

  return {
    controller,
    escapeHtmlEntities,
    genPass,
    guestSalt,
    logoAttachment,
    sendMail,
    testEmail,
  };
}

module.exports = { createEmailService };
