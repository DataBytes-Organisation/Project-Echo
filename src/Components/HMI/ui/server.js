const JWT_SECRET = "deff1952d59f883ece260e8683fed21ab0ad9a53323eca4f";
const express = require('express');
const app = express();
const path = require('path');
const fs = require('fs');
const cookieSession = require('cookie-session');
const helmet = require('helmet');
const jwt = require('jsonwebtoken');
const { client, checkUserSession } = require('./middleware');
const controller = require('./controller/auth.controller');
const crypto = require('crypto');
const bcrypt = require('bcryptjs');
const cors = require('cors');
require('dotenv').config();
const stripe = require('stripe')(process.env.STRIPE_PRIVATE_KEY);
const axios = require('axios');
const { MongoClient, ObjectId } = require('mongodb');
const mongoose = require('mongoose');
const { Server } = require('socket.io');
const http = require('http');
const webpush = require('web-push');
const { User } = require('./model/user.model');
const { Notification, UserNotificationPreference } = require('./model/notification.model');
const bodyParser = require('body-parser');
const Donation = require('./model/donation.model');
const NotificationService = require('./services/notificationService');

// Initialize HTTP server for Socket.io
const server = http.createServer(app);

// Connect to Redis
client.connect();

// MongoDB Connection
mongoose.set('strictQuery', false);

const connectWithRetry = () => {
  mongoose.connect('mongodb://root:root_password@ts-mongodb-cont:27017/EchoNet?authSource=admin', {
    useNewUrlParser: true,
    useUnifiedTopology: true,
    serverSelectionTimeoutMS: 5000,
    socketTimeoutMS: 45000
  })
      .then(() => console.log('MongoDB connection established'))
      .catch(err => {
        console.error('MongoDB connection error:', err);
        setTimeout(connectWithRetry, 5000);
      });
};

connectWithRetry();

const port = 8080;
const rootDirectory = __dirname;

// Initialize Socket.io
const io = new Server(server, {
  cors: {
    origin: [
      "http://localhost:8080",
      "http://localhost:8081",
      "http://127.0.0.1:8080"
    ],
    methods: ["GET", "POST"],
    credentials: true,
    allowedHeaders: ["authorization"]
  },
  transports: ['websocket', 'polling']
});

// Track connected users
const connectedUsers = new Map();

// Socket.io middleware for authentication
const authMiddleware = (req, res, next) => {
  const token = req.headers.authorization?.split(' ')[1];

  if (!token) {
    return res.status(401).json({ error: "No token provided" });
  }

  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    req.user = decoded;
    next();
  } catch (err) {
    console.error("JWT verification failed:", err.message);
    return res.status(401).json({
      error: "Invalid token",
      details: err.message
    });
  }
};

// Socket.io connection handler
io.on('connection', (socket) => {
  console.log(`User connected: ${socket.userId}`);

  // Add to connected users map
  connectedUsers.set(socket.userId, socket.id);

  // Send any pending notifications
  sendPendingNotifications(socket.userId);

  socket.on('disconnect', () => {
    console.log(`User disconnected: ${socket.userId}`);
    connectedUsers.delete(socket.userId);
  });

  // Handle mark as read
  socket.on('markAsRead', async (notificationId) => {
    try {
      const notification = await Notification.findByIdAndUpdate(
          notificationId,
          { status: 'read' },
          { new: true }
      );

      if (notification) {
        socket.emit('notificationUpdated', notification);
      }
    } catch (error) {
      console.error('Error marking notification as read:', error);
    }
  });
});

// Helper function to send pending notifications
async function sendPendingNotifications(userId) {
  try {
    const notifications = await Notification.find({
      userId,
      status: 'unread'
    }).sort({ createdAt: -1 }).limit(50);

    if (notifications.length > 0) {
      const socketId = connectedUsers.get(userId);
      if (socketId) {
        io.to(socketId).emit('initialNotifications', notifications);
      }
    }
  } catch (error) {
    console.error('Error sending pending notifications:', error);
  }
}

// Function to send real-time notification
async function sendRealTimeNotification(userId, notification) {
  const socketId = connectedUsers.get(userId);
  if (socketId) {
    io.to(socketId).emit('newNotification', notification);
  }
  // Also send via other channels
  await NotificationService.dispatchNotification(notification);
}

app.get('/api/notification-preferences', async (req, res) => {
  try {
    const token = req.headers.authorization?.split(' ')[1];
    if (!token) return res.status(401).json({ error: 'Unauthorized' });

    const decoded = jwt.verify(token, JWT_SECRET);
    const preferences = await UserNotificationPreference.findOne({ userId: decoded.id }) ||
        new UserNotificationPreference({ userId: decoded.id });

    res.json(preferences);
  } catch (error) {
    res.status(500).json({ error: 'Error fetching preferences' });
  }
});

app.put('/api/notification-preferences', async (req, res) => {
  try {
    const token = req.headers.authorization?.split(' ')[1];
    if (!token) return res.status(401).json({ error: 'Unauthorized' });

    const decoded = jwt.verify(token, JWT_SECRET);
    const preferences = await UserNotificationPreference.findOneAndUpdate(
        { userId: decoded.id },
        req.body,
        { new: true, upsert: true }
    );

    res.json(preferences);
  } catch (error) {
    res.status(500).json({ error: 'Error updating preferences' });
  }
});

const storeItems = new Map([[
  1, { priceInCents: 100, name: "donation"}
]]);

app.use(express.json({limit: '10mb'}));
app.use(express.static(path.join(__dirname, 'public'), {
  index: path.join(__dirname, 'public/login.html')
}));

var corsOptions = {
  origin: ["http://localhost:8081", "*"]
};

app.use(cors(corsOptions));
app.use(bodyParser.json({ limit: '10mb' }));
app.use(bodyParser.urlencoded({ extended: true, limit: '10mb' }));

app.use(
    cookieSession({
      name: "echo-session",
      keys: ["COOKIE_SECRET"],
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

function escapeHtmlEntities(input) {
  return input.replace(/[\u00A0-\u9999<>&]/gim, function (i) {
    return "&#" + i.charCodeAt(0) + ";";
  });
}

async function testEmail(input) {
  let res = await validation.validate(input);
  return {result: res.valid, response: res.validators};
}

// Middleware to check if the user is an admin
function isAdmin(req, res, next) {
  const token = req.headers.authorization?.split(' ')[1];
  if (!token) {
    return res.status(401).json({ message: 'No token provided' });
  }

  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    if (decoded.role !== 'admin') {
      return res.status(403).json({ message: 'Access denied: Admins only' });
    }
    next();
  } catch (error) {
    return res.status(401).json({ message: 'Invalid token' });
  }
}

// Admin user management APIs
app.patch('/api/users/:id/suspend', isAdmin, async (req, res) => {
  try {
    const user = await User.findByIdAndUpdate(
        req.params.id,
        { status: 'suspended' },
        { new: true }
    );
    res.json({ message: `User ${user.email} suspended`, user });
  } catch (error) {
    res.status(500).json({ error: 'Error suspending user' });
  }
});

app.patch('/api/users/:id/ban', isAdmin, async (req, res) => {
  try {
    const user = await User.findByIdAndUpdate(
        req.params.id,
        { status: 'banned' },
        { new: true }
    );
    res.json({ message: `User ${user.email} banned`, user });
  } catch (error) {
    res.status(500).json({ error: 'Error banning user' });
  }
});

app.patch('/api/users/:id/reinstate', isAdmin, async (req, res) => {
  try {
    const user = await User.findByIdAndUpdate(
        req.params.id,
        { status: 'active' },
        { new: true }
    );
    res.json({ message: `User ${user.email} reinstated`, user });
  } catch (error) {
    res.status(500).json({ error: 'Error reinstating user' });
  }
});

app.get('/api/users/:id/status', isAdmin, async (req, res) => {
  try {
    const user = await User.findById(req.params.id, 'email status');
    res.json({ user });
  } catch (error) {
    res.status(500).json({ error: 'Error retrieving user status' });
  }
});

// Stripe payment endpoints
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

    // Create notification for admin
    const adminUsers = await User.find({ role: 'admin' });
    for (const admin of adminUsers) {
      await Notification.create({
        userId: admin._id,
        title: "New Donation Received",
        message: `Donation from ${req.body.userEmail}`,
        type: "donation",
        link: "/admin-donations"
      });
    }

    res.json({ url: session.url });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get('/donations', async (req, res) => {
  try {
    const donations = await Donation.find({});
    res.json({ charges: { data: donations } });
  } catch (error) {
    console.error("Error fetching donations:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.get('/cumulativeDonations', async(req, res) => {
  try {
    const donations = await Donation.find({});
    const cumulativeTotal = donations.reduce((sum, donation) => sum + donation.amount, 0) / 100;
    res.json({ cumulativeTotal: cumulativeTotal.toFixed(2) });
  } catch(error) {
    console.error('Error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Email endpoints
app.post("/send_email", async (req, res) => {
  const { email, query } = req.body;
  const validationResult = await testEmail(email);

  if (validationResult.result) {
    let html_text = '<div>';
    html_text += '<h2>A new query has been received for Project Echo HMI</h2>';
    html_text += '<img src="cid:logo@echo.hmi" style="height: 150px; width: 150px; display: flex; margin: auto;"/>';
    html_text += '<p>Sender: \t ' + email + '</p>';
    html_text += '<p>Query: \t ' + escapeHtmlEntities(query) + '</p>';
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
        res.status(500).send('<script>alert("Failed to send email"); window.location.href = "/";</script>');
      } else {
        console.log('Email sent: ' + info.response);
        res.send('<script>alert("User query sent! Please check your mailbox"); window.location.href = "/";</script>');
      }
    });
  } else {
    res.status(400).send('<script>alert("Invalid email address"); window.location.href = "/";</script>');
  }
});

// Notification endpoints
app.get('/api/notifications', async (req, res) => {
  try {
    const token = req.headers.authorization?.split(' ')[1];
    if (!token) return res.status(401).json({ error: 'Unauthorized' });

    const decoded = jwt.verify(token, JWT_SECRET);

    // Add validation for limit/offset
    const limit = Math.min(parseInt(req.query.limit) || 20, 100);
    const offset = parseInt(req.query.offset) || 0;

    const notifications = await Notification.find({ userId: decoded.id })
        .sort({ createdAt: -1 })
        .skip(offset)
        .limit(limit);

    res.json({ notifications });
  } catch (error) {
    console.error('Notification fetch error:', error);
    res.status(500).json({
      error: 'Error fetching notifications',
      details: error.message
    });
  }
});

app.patch('/api/notifications/:id/read', async (req, res) => {
  try {
    const token = req.headers.authorization?.split(' ')[1];
    if (!token) return res.status(401).json({ error: 'Unauthorized' });

    const decoded = jwt.verify(token, JWT_SECRET);

    const notification = await Notification.findOneAndUpdate(
        { _id: req.params.id, userId: decoded.id },
        { status: 'read' },
        { new: true }
    );

    if (!notification) {
      return res.status(404).json({ error: 'Notification not found' });
    }

    res.json(notification);
  } catch (error) {
    res.status(500).json({ error: 'Error updating notification' });
  }
});

// Other routes
require('./routes/auth.routes')(app);
require('./routes/user.routes')(app);
require('./routes/map.routes')(app);

// Route handlers
app.get('*', checkUserSession);

app.get("/verify-otp", (req, res) => {
  res.sendFile(path.join(__dirname, 'public/verify-otp.html'));
});

app.get("/", async (req, res) => {
  try {
    const role = await client.get('Roles');
    if (role?.toLowerCase().includes("admin")) {
      res.redirect("/admin-dashboard");
    } else {
      res.redirect("/map");
    }
  } catch (error) {
    res.redirect("/login");
  }
});

app.get("/welcome", async (req, res) => {
  try {
    const role = await client.get('Roles');
    if (role?.toLowerCase().includes("admin")) {
      res.redirect("/admin-dashboard");
    } else {
      res.redirect("/map");
    }
  } catch {
    res.send('<script>alert("No user info detected! Please login again"); window.location.href = "/login";</script>');
  }
});

// Temporary test route - REMOVE AFTER TESTING
app.post('/test-notification', async (req, res) => {
  try {
    const notification = await Notification.create({
      userId: req.body.userId, // or a fixed admin user ID
      title: req.body.title || "Test Notification",
      message: req.body.message || "This is a test notification",
      type: "donation",
      status: "unread"
    });

    // Send real-time update if user is connected
    sendRealTimeNotification(notification.userId, notification);

    res.json(notification);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Admin routes
app.get("/admin-dashboard", (req, res) => {
  res.sendFile(path.join(__dirname, 'public/admin/dashboard.html'));
});

app.get("/admin-nodes", (req, res) => {
  res.sendFile(path.join(__dirname, 'public/admin/admin-nodes.html'));
});

app.get("/admin-profile", (req, res) => {
  res.sendFile(path.join(__dirname, 'public/admin/profile.html'));
});

app.get("/admin-template", (req, res) => {
  res.sendFile(path.join(__dirname, 'public/admin/template.html'));
});

app.get("/admin-donations", (req, res) => {
  res.sendFile(path.join(__dirname, 'public/admin/donations.html'));
});

app.get("/admin-notifications", (req, res) => {
  res.sendFile(path.join(__dirname, 'public/admin/notifications.html'));
});

app.get("/login", (req, res) => {
  res.sendFile(path.join(__dirname, 'public/login.html'));
});

app.get("/map", (req, res) => {
  res.sendFile(path.join(__dirname, 'public/index.html'));
});

// Error handling
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).send('Something broke!');
});

process.on('unhandledRejection', (err) => {
  console.error('Unhandled Rejection:', err);
});

process.on('uncaughtException', (err) => {
  console.error('Uncaught Exception:', err);
});

// Start the server
server.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});