const express = require('express');
const path = require('path');
const cookieSession = require('cookie-session');
const helmet = require('helmet');
const crypto = require('crypto');
const cors = require('cors');
const axios = require('axios');
const { MongoClient } = require('mongodb');
const controller = require('../controller/auth.controller');
const { User } = require('./services/user.model');
const { client, createApiSessionGuard, checkUserSession: defaultCheckUserSession, resolveLandingPath, requireApiSession: defaultRequireApiSession } = require('./middleware');
const { createCheckUserSession } = require('./middleware/session');
const apiClient = require('./services/apiClient');
const { createEmailService } = require('./services/email');
const { createNotificationStore } = require('./services/notificationStore');
const adminRoutes = require('./routes/admin.routes');
const contactRoutes = require('./routes/contact.routes');
const legacyPaymentRoutes = require('./routes/legacy-payments.routes');
const notificationsRoutes = require('./routes/notifications.routes');
const pagesRoutes = require('./routes/pages.routes');
const proxyRoutes = require('./routes/proxy.routes');
const razorpayPayment = require('./routes/razorpay.routes');

const rootDirectory = path.resolve(__dirname, '..');
const publicDir = path.join(rootDirectory, 'public');
const API_BASE_URL = apiClient.API_BASE_URL;

function createApp(dependencies = {}) {
  const app = express();
  const donationClient = dependencies.donationClient || dependencies.databases?.donationClient || new MongoClient(process.env.MONGODB_URI || `mongodb://${process.env.DB_HOST || 'localhost'}:27017`, {
    useNewUrlParser: true,
    useUnifiedTopology: true
  });
  const dbState = { connectedDB: dependencies.connectedDB || dependencies.databases?.echoNetDb };
  const redisClient = dependencies.redisClient || client;
  const checkUserSession = dependencies.checkUserSession || (dependencies.redisClient ? createCheckUserSession(redisClient) : defaultCheckUserSession);
  const requireApiSession = dependencies.requireApiSession || (dependencies.redisClient ? createApiSessionGuard(async () => redisClient.get('JWT')) : defaultRequireApiSession);
  const stripe = dependencies.stripe || require('stripe')(process.env.STRIPE_PRIVATE_KEY);
  const storeItems = dependencies.storeItems || new Map([[
    1, { priceInCents: 100, name: 'donation' }
  ]]);
  const apiBaseUrl = dependencies.config
    ? apiClient.resolveApiBaseUrl({ API_HOST: dependencies.config.apiHost, API_PORT: dependencies.config.apiPort })
    : API_BASE_URL;
  const notificationStore = dependencies.notificationStore || createNotificationStore({ donationClient, dbState });
  const emailService = dependencies.emailService || createEmailService({ rootDirectory, controller });

  // Razorpay webhook needs raw bytes before JSON parsing.
  app.post('/api/razorpay-webhook', express.raw({ type: 'application/json' }), (req, res) => {
    return razorpayPayment.handleWebhook(req, res, { apiBaseUrl });
  });

  app.use(express.json({ limit: '10mb' }));
  const cookieSecret = process.env.COOKIE_SECRET || crypto.randomBytes(32).toString('hex');
  // ponytail: ephemeral fallback invalidates sessions on restart; configure COOKIE_SECRET for stable sessions.
  app.use(
    cookieSession({
      name: 'echo-session',
      keys: [cookieSecret],
      httpOnly: true,
      sameSite: 'lax'
    })
  );

  adminRoutes.registerRoutes(app, { apiClient, redisClient, User });

  app.use(
    helmet({
      contentSecurityPolicy: {
        useDefaults: true,
        directives: razorpayPayment.withRazorpayCsp({
          defaultSrc: ["'self'"],
          scriptSrc: [
            "'self'",
            "'unsafe-inline'",
            'https://cdn.jsdelivr.net',
            'https://cdnjs.cloudflare.com',
            'https://code.jquery.com',
            'https://kit.fontawesome.com',
            'https://www.google.com',
            'https://www.gstatic.com',
            'https://www.recaptcha.net'
          ],
          scriptSrcElem: [
            "'self'",
            "'unsafe-inline'",
            'https://cdn.jsdelivr.net',
            'https://cdnjs.cloudflare.com',
            'https://code.jquery.com',
            'https://kit.fontawesome.com',
            'https://www.google.com',
            'https://www.gstatic.com',
            'https://www.recaptcha.net'
          ],
          scriptSrcAttr: ["'unsafe-inline'"],
          styleSrc: [
            "'self'",
            "'unsafe-inline'",
            'https://cdn.jsdelivr.net',
            'https://fonts.googleapis.com',
            'https://cdnjs.cloudflare.com'
          ],
          styleSrcElem: [
            "'self'",
            "'unsafe-inline'",
            'https://cdn.jsdelivr.net',
            'https://fonts.googleapis.com',
            'https://cdnjs.cloudflare.com'
          ],
          fontSrc: [
            "'self'",
            'https://fonts.gstatic.com',
            'https://cdn.jsdelivr.net',
            'https://cdnjs.cloudflare.com',
            'https://kit.fontawesome.com',
            'https://ka-f.fontawesome.com',
            'data:'
          ],
          imgSrc: [
            "'self'",
            'data:',
            'blob:',
            'https:'
          ],
          mediaSrc: [
            "'self'",
            'blob:'
          ],
          connectSrc: [
            "'self'",
            'ws:',
            'wss:',
            'http://localhost:3000',
            'http://localhost:9000',
            'http://localhost:8000',
            'https://cdn.jsdelivr.net',
            'https://cdnjs.cloudflare.com',
            'https://www.google.com',
            'https://www.gstatic.com',
            'https://www.recaptcha.net',
            'https://ka-f.fontawesome.com'
          ],
          frameSrc: [
            "'self'",
            'https://js.stripe.com',
            'https://www.google.com',
            'https://www.recaptcha.net'
          ],
          objectSrc: ["'none'"],
          upgradeInsecureRequests: null
        })
      }
    })
  );

  legacyPaymentRoutes.registerRoutes(app, { stripe, storeItems, donationClient, dbState, clientUrl: dependencies.config?.clientUrl || process.env.CLIENT_URL || 'http://localhost:3000' });

  razorpayPayment.registerRazorpayBrowserRoutes(app, {
    apiBaseUrl,
    checkUserSession,
  });

  app.get('/index.html', checkUserSession);
  app.use('/pages/map', checkUserSession);
  app.use('/pages/admin', checkUserSession);
  app.use(express.static(publicDir, { index: path.join(publicDir, 'pages/auth/login.html') }));

  app.use(cors({ origin: ['http://localhost:8081', '*'] }));
  app.use(express.json({ limit: '10mb' }));
  app.use(express.urlencoded({ extended: true, limit: '10mb' }));

  contactRoutes.registerRoutes(app, { emailService });

  require('./routes/auth.routes')(app);
  require('./routes/user.routes')(app);
  require('./routes/map.routes')(app);
  // Sensor-specific routes must stay before the broad /sensors proxy.
  require('./routes/sensor.routes')(app, { apiBaseUrl });

  pagesRoutes.registerRoutes(app, {
    publicDir,
    checkUserSession,
    redisClient,
    resolveLandingPath,
  });

  notificationsRoutes.registerRoutes(app, {
    requireApiSession,
    notificationStore,
  });

  proxyRoutes.registerRoutes(app, {
    axios,
    apiClient,
    apiBaseUrl,
    checkUserSession,
  });

  return app;
}

module.exports = { createApp };
