const express = require('express');
const app = express();
require('dotenv').config(); // Load environment variables
const path = require('path');
const cookieSession = require('cookie-session');
const helmet = require('helmet');
const cors = require('cors');


// Import routes
const authRoutes = require('./routes/auth.routes');
const mapRoutes = require('./routes/map.routes');
const userRoutes = require('./routes/user.routes');
const paymentRoutes = require('./routes/payment.routes');
const emailRoutes = require('./routes/email.routes');

// Middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(helmet()); // Security middleware

// Session management
app.use(
  cookieSession({
    name: "echo-session",
    keys: [process.env.COOKIE_SECRET_KEY], // Use secret from .env
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    maxAge: 24 * 60 * 60 * 1000, // 24 hours
  })
);

// Enable CORS
const corsOptions = {
  origin: [process.env.CLIENT_SERVER_URL, "*"]
};
app.use(cors(corsOptions));

// Use the routes
app.use('/api/auth', authRoutes); // Base path added for auth routes
app.use('/api/map', mapRoutes); // Base path added for map routes
app.use('/api/user', userRoutes); // Base path added for user routes
app.use('/api/payment', paymentRoutes); // Base path added for payment routes
app.use('/api/email', emailRoutes); // Base path added for email routes

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));

// Handle the root URL explicitly
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public/login.html'));
});

// Default route to handle all other requests
app.get('*', (req, res) => {
  res.status(404).send('Page Not Found');
});

// Start the server
const port = process.env.PORT || 8080;
app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
