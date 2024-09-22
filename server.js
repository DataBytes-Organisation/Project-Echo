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
app.use(express.static(path.join(__dirname, 'public')));

// Serve the animal sighting form (index.html) when visiting the root URL
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Cookie session setup
app.use(cookieSession({
    name: "echo-session",
    keys: [process.env.COOKIE_SECRET], // Use environment variable for secret
    httpOnly: true,
}));

const validation = require('deep-email-validator');

// Initialize requests.json if it doesn't exist
const filePath = path.join(__dirname,  'requests.json');
if (!fs.existsSync(filePath)) {
    fs.writeFileSync(filePath, JSON.stringify([]), 'utf8');
}

// Helper function to sanitize inputs
function escapeHtmlEntities(input) {
    return input.replace(/[\u00A0-\u9999<>&]/gim, function (i) {
        return "&#" + i.charCodeAt(0) + ";";
    });
}

// Nodemailer transporter setup using environment variables
const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
        user: process.env.EMAIL_USER,
        pass: process.env.EMAIL_PASS,
    },
});

// Animal Sighting Submission API
app.post('/api/sightings', (req, res) => {
    const { species, location, dateTime, observations } = req.body;

    // Log the incoming data to the console
    console.log('Species:', species);
    console.log('Location:', location);
    console.log('Date/Time:', dateTime);
    console.log('Observations:', observations);

    // Input validation
    if (!species || !location || !dateTime || !location.latitude || !location.longitude) {
        return res.status(400).json({ message: 'All required fields must be filled' });
    }

    if (isNaN(location.latitude) || location.latitude < -90 || location.latitude > 90 || isNaN(location.longitude) || location.longitude < -180 || location.longitude > 180) {
        return res.status(400).json({ message: 'Invalid latitude or longitude' });
    }

    // Sanitize user inputs
    const sanitizedSpecies = escapeHtmlEntities(species);
    const sanitizedObservations = escapeHtmlEntities(observations || '');

    // Read requests file and save the new sighting
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

        const newSighting = {
            species: sanitizedSpecies,
            location: {
                latitude: parseFloat(location.latitude),
                longitude: parseFloat(location.longitude)
            },
            dateTime,
            observations: sanitizedObservations,
        };

        requests.push(newSighting);

        fs.writeFile(filePath, JSON.stringify(requests, null, 2), err => {
            if (err) {
                return res.status(500).json({ message: 'Failed to save request' });
            }

            // Log the successful submission
            console.log('New sighting submitted:', newSighting);
            res.json({ message: 'Sighting submitted successfully', sighting: newSighting });
        });
    });
});

// API to retrieve all animal sightings
app.get('/api/sightings', (req, res) => {
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

        res.status(200).json(requests);
    });
});


// Handle email submission
app.post("/send_email", async (req, res) => {
    const { email, query } = req.body;
    const validationResult = await validation.validate(email);
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
                return res.status(500).json({ message: 'Failed to send email' });
            } else {
                console.log('Email sent: ' + info.response);
                return res.send('<script> alert("User query sent! Please check your mailbox for further communication"); window.location.href = "/"; </script>');
            }
        });
    } else {
        return res.status(400).send("<script> alert('Sender\'s email is not valid!')</script>");
    }
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
