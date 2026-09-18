function registerRoutes(app, dependencies) {
  const { emailService } = dependencies;

  app.post('/send_email', async (req, res) => {
    const { email, query } = req.body;
    const validationResult = await emailService.testEmail(email);
    if (validationResult.result) {
      let htmlText = '<div>';
      htmlText += '<h2>A new query has been received for Project Echo HMI</h2>';
      htmlText += '<img src="cid:logo@echo.hmi" style="height: 150px; width: 150px; display: flex; margin: auto;"/>';
      htmlText += '<p>Sender: \t ' + email + '</p>';
      htmlText += '<p>Query: \t ' + emailService.escapeHtmlEntities(query) + '</p>';
      htmlText += '<hr>';
      htmlText += '<p>Yes, this mailbox is active. So please feel free to reply to this email if you have other queries.</p>';
      htmlText += '</div>';

      let mailOptions = {
        from: email,
        to: `echodatabytes@gmail.com, ${email}`,
        subject: 'New query received!',
        text: query,
        html: htmlText,
        attachments: emailService.logoAttachment()
      };

      emailService.sendMail(mailOptions, function (error, info) {
        if (error) {
          console.log(error);
        } else {
          console.log('Email sent: ' + info.response);
          return res.send('<script> alert("user query sent! Please check your mailbox for further communication"); window.location.href = "/"; </script>');
        }
      });
    } else {
      return res.status(400).send('<script> alert(`Sender\'s email is not valid!`)</script>');
    }
  });

  app.get('/send_email', (req, res) => {
    setTimeout(() => res.redirect('/'), 5000);
  });

  app.post('/request_access', async (req, res) => {
    console.log('email: ', req.body.email);
    const { email } = req.body;
    let salt = emailService.guestSalt();
    let username = 'guest_' + email.split('@')[0] + '_' + salt;
    let password = emailService.genPass(12);
    let timestamp = new Date(Date.now() + 1800000);
    let request = {
      username: username,
      email: req.body.email,
      password: password,
      timestamp: timestamp
    };
    console.log('Guest details: ', request);
    try {
      const response = await emailService.controller.guestsignup(request);

      setTimeout(() => {
        console.log('response is back! ', response);
        if (response && response.status === 'success') {
          let htmlText = '<div>';
          htmlText += '<h2>Echo HMI Temporary Access Requested!</h2>';
          htmlText += '<img src="cid:logo@echo.hmi" style="height: 150px; width: 150px; display: flex; margin: auto;"/>';
          htmlText += '<p>Dear \t <strong>' + req.body.email + '</strong></p>';
          htmlText += '<hr>';
          htmlText += '<p>Thank you for your patience, here is your login credential </p>';
          htmlText += '<p><strong>Username:</strong> \t ' + username + '</p>';
          htmlText += '<p><strong>Password:</strong> \t ' + password + '</p>';
          htmlText += '<br><p>Please take in mind that this account will only be valid until ' + timestamp.toString() + ' (Subject to change based on development)</p>';
          htmlText += '</div>';
          let mailOptions = {
            from: email,
            to: `echodatabytes@gmail.com, ${email}`,
            subject: 'Guest User Access Granted!',
            html: htmlText,
            attachments: emailService.logoAttachment()
          };
          emailService.sendMail(mailOptions, function (error, info) {
            if (error) {
              console.log(error);
            } else {
              console.log('Email sent: ' + info.response);
              return res.send('<script> alert("Temporary credential granted! Please check your mailbox."); window.location.href = "/login"; </script>');
            }
          });
        } else {
          console.log('Something happened for Guest Access Granting: ', response);
          let error_box = document.getElementById('request-access-email-error');
          error_box.innerHTML = `Exception error occured: ${response.message}`;
          error_box.style.display = 'block';
          setTimeout(() => {
            error_box.innerHTML = '';
            error_box.style.display = 'none';
          }, 3000);
        }

      }, 200);

    } catch (error) {
      res.status(500).send({ message: 'An error occurred while sending the request access: ' + error });
    }
  });
}

module.exports = { registerRoutes };
