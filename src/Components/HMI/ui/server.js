const express = require('express');
const path = require('path');
const fs = require('fs');
const cookieSession = require('cookie-session');
const helmet = require('helmet');
const jwt = require('jsonwebtoken');
const crypto = require('crypto');
const bcrypt = require('bcryptjs');
const cors = require('cors');
const axios = require('axios');
require('dotenv').config();

const app = express();
const port = process.env.PORT || 8080; // Use port from .env or default to 8080

// Middleware and Security
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(cors({ origin: ["http://localhost:8081", "*"] }));
app.use(
  cookieSession({
    name: "echo-session",
    keys: [process.env.COOKIE_SECRET_KEY || "default_cookie_secret"], // Use secret from .env or default
    httpOnly: true
  })
);
// Uncomment to use helmet middleware for security headers
// app.use(helmet());

// Initialize Redis client
const { client, checkUserSession } = require('./middleware');
client.connect();

// Stripe Initialization
const stripe = require('stripe')(process.env.STRIPE_PRIVATE_KEY);

// Setup nodemailer for email functionality
const nodemailer = require('nodemailer');
const transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: process.env.USER_EMAIL || 'default_email@gmail.com',
    pass: process.env.PASSWORD || 'default_password'
  }
});

// Utility Functions
function escapeHtmlEntities(input) {
  return input.replace(/[\u00A0-\u9999<>&]/gim, function (i) {
    return "&#" + i.charCodeAt(0) + ";";
  });
}

async function testEmail(input) {
  const validation = require('deep-email-validator');
  let res = await validation.validate(input);
  return { result: res.valid, response: res.validators };
}

// Routes: Static Files
app.use(express.static(path.join(__dirname, 'public'), { index: path.join(__dirname, 'public/login.html') }));

// Routes: Payment and Donation
app.post("/api/create-checkout-session", async (req, res) => {
  try {
    const session = await stripe.checkout.sessions.create({
      submit_type: 'donate',
      customer_email: req.body.userEmail,
      payment_method_types: ["card"],
      mode: "payment",
      line_items: req.body.items.map(item => {
        const storeItem = storeItems.get(item.id);
        return {
          price_data: {
            currency: "aud",
            product_data: { name: storeItem.name },
            unit_amount: item.quantity * storeItem.priceInCents,
          },
          quantity: 1,
        };
      }),
      success_url: process.env.CLIENT_SERVER_URL,
      cancel_url: process.env.CLIENT_SERVER_URL
    });
    res.json({ url: session.url });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get('/donations', async (req, res) => {
  try {
    let charges;
    let firstPage = false;
    while (true) {
      if (!firstPage) {
        charges = await stripe.charges.list({ limit: 100 });
        firstPage = true;
      }
      if (!charges.has_more) break;

      const nextPage = charges.data[charges.data.length - 1].id;
      charges = await stripe.charges.list({ limit: 100, starting_after: nextPage });
    }
    res.json({ charges });
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/cumulativeDonations', async (req, res) => {
  try {
    let cumulativeTotal = 0;
    let firstPage = false;
    while (true) {
      let charges;
      if (!firstPage) {
        charges = await stripe.charges.list({ limit: 100 });
        firstPage = true;
      }

      charges.data.forEach(charge => {
        cumulativeTotal += charge.amount;
      });

      if (!charges.has_more) break;

      const nextPage = charges.data[charges.data.length - 1].id;
      charges = await stripe.charges.list({ limit: 100, starting_after: nextPage });
    }

    cumulativeTotal = (cumulativeTotal / 100).toFixed(2);
    res.json({ cumulativeTotal });
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Routes: Email
app.post("/send_email", async (req, res) => {
  const { email, query } = req.body;
  const validationResult = await testEmail(email);
  if (validationResult.result) {
    let html_text = `
      <div>
        <h2>A new query has been received for Project Echo HMI</h2>
        <img src="cid:logo@echo.hmi" style="height: 150px; width: 150px; display: flex; margin: auto;"/>
        <p>Sender: \t ${escapeHtmlEntities(email)}</p>
        <p>Query: \t ${escapeHtmlEntities(query)}</p>
        <hr>
        <p>Yes, this mailbox is active. So please feel free to reply to this email if you have other queries.</p>
      </div>
    `;

    let mailOptions = {
      from: email,
      to: `${process.env.USER_EMAIL}, ${email}`,
      subject: 'New query received!',
      text: query,
      html: html_text,
      attachments: [{ 
        filename: 'image.png',
        content: fs.createReadStream(path.join(__dirname, 'public/images/tabIcons/logo.png')),
        cid: 'logo@echo.hmi'
      }]
    };

    transporter.sendMail(mailOptions, function (error, info) {
      if (error) {
        console.log(error);
      } else {
        console.log('Email sent: ' + info.response);
        res.send('<script> alert("user query sent! Please check your mailbox for further communication"); window.location.href = "/"; </script>');
      }
    });
  } else {
    res.status(400).send("<script> alert(`Sender's email is not valid!`)</script>");
  }
});

app.get("/send_email", (req, res) => {
  setTimeout(() => res.redirect("/"), 5000);
});

// Routes: User Access Request
app.post("/request_access", async (req, res) => {
  console.log("email: ", req.body.email);
  const { email } = req.body;

  let salt = crypto.randomBytes(8).toString('hex');
  let username = `guest_${email.split('@')[0]}_${salt}`;
  let password = genPass(12);
  let timestamp = new Date(Date.now() + 1800000); // 30 minutes TTL

  let request = { username, email, password, timestamp };
  console.log("Guest details: ", request);

  try {
    const response = await controller.guestsignup(request);

    if (response && response.status === 'success') {
      let html_text = `
        <div>
          <h2>Echo HMI Temporary Access Requested!</h2>
          <img src="cid:logo@echo.hmi" style="height: 150px; width: 150px; display: flex; margin: auto;"/>
          <p>Dear <strong>${email}</strong></p>
          <hr>
          <p>Thank you for your patience, here is your login credential:</p>
          <p><strong>Username:</strong> ${username}</p>
          <p><strong>Password:</strong> ${password}</p>
          <br>
          <p>Please note that this account will only be valid until ${timestamp.toString()} (Subject to change based on development)</p>
        </div>
      `;

      let mailOptions = {
        from: email,
        to: `${process.env.USER_EMAIL}, ${email}`,
        subject: 'Guest User Access Granted!',
        html: html_text,
        attachments: [{
          filename: 'image.png',
          content: fs.createReadStream(path.join(__dirname, 'public/images/tabIcons/logo.png')),
          cid: 'logo@echo.hmi'
        }]
      };

      transporter.sendMail(mailOptions, function (error, info) {
        if (error) {
          console.log(error);
        } else {
          console.log('Email sent: ' + info.response);
          res.send('<script> alert("Temporary credential granted! Please check your mailbox."); window.location.href = "/login"; </script>');
        }
      });
    } else {
      console.log("Something happened for Guest Access Granting: ", response);
      res.status(500).send({ message: 'Failed to grant access.' });
    }

  } catch (error) {
    res.status(500).send({ message: 'An error occurred while sending the request access: ' + error });
  }
});

// Routes: Other App Routes
require('./routes/auth.routes')(app);
require('./routes/user.routes')(app);
require('./routes/map.routes')(app);

// Routes: Admin and General Pages
app.get('/admin-dashboard', (req, res) => res.sendFile(path.join(__dirname, 'public/admin/dashboard.html')));
app.get('/admin-template', (req, res) => res.sendFile(path.join(__dirname, 'public/admin/template.html')));
app.get('/admin-donations', (req, res) => res.sendFile(path.join(__dirname, 'public/admin/donations.html')));
app.get('/login', (req, res) => res.sendFile(path.join(__dirname, 'public/login.html')));

app.get('*', checkUserSession);

// API Routes: Requests
app.post("/api/submit", async (req, res) => {
  let token = await client.get('JWT').catch(err => console.error('Error retrieving token from Redis:', err));
  let schema = req.body;
  schema.status = "pending";
  schema.date = new Date();

  try {
    const axiosResponse = await axios.post('http://ts-api-cont:9000/hmi/api/submit', schema, { headers: { "Authorization": `Bearer ${token}`, 'Content-Type': 'application/json' } });
    if (axiosResponse.status === 201) {
      res.status(201).send('<script> window.location.href = "/login"; alert("Request Submitted successfully");</script>');
    } else {
      res.status(400).send('<script> window.location.href = "/login"; alert("Oops! Something went wrong");</script>');
    }
  } catch (error) {
    console.error(error.data);
    res.status(500).send("An error occurred");
  }
});

app.patch('/api/requests/:id', async (req, res) => {
  const { id: requestId } = req.params;
  const { status: newStatus } = req.body;

  let token = await client.get('JWT').catch(err => console.error('Error retrieving token from Redis:', err));
  let schema = { requestId, newStatus };

  try {
    const axiosResponse = await axios.patch('http://ts-api-cont:9000/hmi/api/requests', schema, { headers: { "Authorization": `Bearer ${token}`, 'Content-Type': 'application/json' } });
    if (axiosResponse.status === 200) {
      res.status(200).send('<script> window.location.href = "/login"; alert("Request data updated successfully");</script>');
    } else {
      res.status(400).send('<script> window.location.href = "/login"; alert("Oops! Something went wrong with updating request table");</script>');
    }
  } catch (error) {
    console.error(error.data);
    res.status(500).send({ error: 'Error updating request status' });
  }
});

app.patch('/api/updateConservationStatus/:animal', async (req, res) => {
  const { animal: requestAnimal } = req.params;
  const { status: newStatus } = req.body;

  let token = await client.get('JWT').catch(err => console.error('Error retrieving token from Redis:', err));
  let schema = { requestAnimal, newStatus };

  try {
    const axiosResponse = await axios.patch('http://ts-api-cont:9000/hmi/api/updateConservationStatus', schema, { headers: { "Authorization": `Bearer ${token}`, 'Content-Type': 'application/json' } });
    if (axiosResponse.status === 200) {
      res.status(200).send('<script> window.location.href = "/login"; alert("Species Data updated successfully");</script>');
    } else {
      res.status(400).send('<script> window.location.href = "/login"; alert("Oops! Something went wrong with updating species data");</script>');
    }
  } catch (error) {
    console.error(error.data);
    res.status(500).send({ error: 'Error updating species status' });
  }
});

// Fetch requests for the admin dashboard
app.get('/api/requests', async (req, res) => {
  try {
    let token = await client.get('JWT').catch(err => console.error('Error retrieving token from Redis:', err));
    const axiosResponse = await axios.get('http://ts-api-cont:9000/hmi/requests', { headers: { "Authorization": `Bearer ${token}` } });

    if (axiosResponse.status === 200) {
      res.json(axiosResponse.data);
    } else {
      res.status(500).json({ error: 'Error fetching data' });
    }
  } catch (err) {
    console.error('Requests error: ', err);
    res.status(401).redirect('/admin-dashboard');
  }
});

// Page Redirections
app.get("/welcome", async (req, res) => {
  try {
    let token = await client.get('JWT').catch(err => console.error('Error retrieving token from Redis:', err));
    let role = await client.get('Roles').catch(err => console.error('Error retrieving user role from Redis:', err));

    if (role.toLowerCase().includes("admin")) {
      res.redirect("/admin-dashboard");
    } else {
      res.redirect("/map");
    }
  } catch {
    res.send('<script> alert("No user info detected! Please login again"); window.location.href = "/login"; </script>');
  }
});

app.get("/map", (req, res) => res.sendFile(path.join(__dirname, 'public/index.html')));

// Start the server
app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});

// Utility Functions
function genPass(length) {
  let password = "";
  const chars = "0123456789abcdefghijklmnopqrstuvwxyz!@#$%^&*()ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  for (let i = 0; i <= parseInt(length); i++) {
    let randomNumber = Math.floor(Math.random() * chars.length);
    password += chars.substring(randomNumber, randomNumber + 1);
  }
  return password;
}
