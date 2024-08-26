const express = require('express');
const fs = require('fs');
const path = require('path');
const cookieSession = require('cookie-session');
const helmet = require('helmet');
const jwt = require('jsonwebtoken');
const crypto = require('crypto');
const bcrypt = require('bcryptjs');
const cors = require('cors');
const nodemailer = require('nodemailer');
const stripe = require('stripe')(process.env.STRIPE_PRIVATE_KEY);
const axios = require('axios');
require('dotenv').config();

const app = express();
const port = 8080;
const rootDirectory = __dirname;

// Middleware to parse JSON
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(cors({ origin: ["http://localhost:8081"] }));
app.use(helmet({
    contentSecurityPolicy: false,
}));

// Middleware to serve static files
app.use(express.static(path.join(__dirname, 'public'), { index: path.join(__dirname, 'public/login.html') }));

// Cookie session setup
app.use(cookieSession({
    name: "echo-session",
    keys: [process.env.COOKIE_SECRET], // Use environment variable for secret
    httpOnly: true,
}));

const validation = require('deep-email-validator');

// Initialize requests.json if it doesn't exist
const filePath = path.join(__dirname, 'requests.json');
if (!fs.existsSync(filePath)) {
    fs.writeFileSync(filePath, JSON.stringify([]), 'utf8');
}

function escapeHtmlEntities(input) {
    return input.replace(/[\u00A0-\u9999<>&]/gim, function (i) {
        return "&#" + i.charCodeAt(0) + ";";
    });
}

function genPass(length) {
    const chars = "0123456789abcdefghijklmnopqrstuvwxyz!@#$%^&*()ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let password = "";
    for (let i = 0; i <= parseInt(length); i++) {
        let randomNumber = Math.floor(Math.random() * chars.length);
        password += chars.substring(randomNumber, randomNumber + 1);
    }
    return password;
}

async function testEmail(input) {
    let res = await validation.validate(input);
    return { result: res.valid, response: res.validators };
}

// Nodemailer transporter setup using environment variables
const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
        user: process.env.EMAIL_USER,
        pass: process.env.EMAIL_PASS,
    },
});

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'notification_form.html'));
});

app.get('/success', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'submission-success.html'));
});

app.post('/submit-request', (req, res) => {
    const requestData = req.body;

    fs.readFile(filePath, 'utf8', (err, data) => {
        if (err) {
            return res.status(500).json({ message: 'Failed to read requests file' });
        }

        let requests;
        try {
            requests = JSON.parse(data);
        } catch (parseError) {
            return res.status(500).json({ message: 'Failed to parse requests file' });
        }

        requests.push(requestData);

        fs.writeFile(filePath, JSON.stringify(requests, null, 2), err => {
            if (err) {
                return res.status(500).json({ message: 'Failed to save request' });
            }
            res.json({ message: 'Request submitted successfully' });
        });
    });
});

app.get('/requests', (req, res) => {
    fs.readFile(filePath, 'utf8', (err, data) => {
        if (err) {
            return res.status(500).json({ message: 'Failed to read requests file' });
        }

        let requests;
        try {
            requests = JSON.parse(data);
        } catch (parseError) {
            return res.status(500).json({ message: 'Failed to parse requests file' });
        }

        requests = requests.filter(request => request && typeof request.username === 'string');

        res.json(requests);
    });
});

app.post("/send_email", async (req, res) => {
    const { email, query } = req.body;
    const validationResult = await testEmail(email);
    if (validationResult.result) {
        let html_text = '<div>';
        html_text += '<h2>A new query has been received for Project Echo HMI</h2>';
        html_text += '<img src="cid:logo@echo.hmi" style="height: 150px; width: 150px; display: flex; margin: auto;"/>';
        html_text += `<p>Sender: \t ${email}</p>`;
        html_text += `<p>Query: \t ${escapeHtmlEntities(query)}</p>`;
        html_text += '<hr>';
        html_text += '<p>Yes, this mailbox is active. So please feel free to reply to this email if you have other queries.</p>';
        html_text += '</div>';

        let mailOptions = {
            from: email,
            to: `echodatabytes@gmail.com, ${email}`,
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
                return res.send('<script> alert("User query sent! Please check your mailbox for further communication"); window.location.href = "/"; </script>');
            }
        });
    } else {
        return res.status(400).send("<script> alert('Sender\'s email is not valid!')</script>");
    }
});

app.get('/welcome', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

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
                        product_data: {
                            name: storeItem.name,
                        },
                        unit_amount: item.quantity * 100,
                    },
                    quantity: 1,
                };
            }),
            success_url: "http://localhost:8080",
            cancel_url: "http://localhost:8080"
        });
        res.json({ url: session.url });
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
