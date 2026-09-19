const razorpayPayment = require("../routes/razorpay.routes");

function createCookieSessionOptions(config) {
  return {
    name: "echo-session",
    keys: [config.cookieSecret],
    httpOnly: true,
    sameSite: "lax",
  };
}

function createHelmetOptions() {
  return {
    contentSecurityPolicy: {
      useDefaults: true,
      directives: razorpayPayment.withRazorpayCsp({
        defaultSrc: ["'self'"],
        scriptSrc: ["'self'", "'unsafe-inline'", "https://cdn.jsdelivr.net", "https://cdnjs.cloudflare.com", "https://code.jquery.com", "https://kit.fontawesome.com", "https://www.google.com", "https://www.gstatic.com", "https://www.recaptcha.net"],
        scriptSrcElem: ["'self'", "'unsafe-inline'", "https://cdn.jsdelivr.net", "https://cdnjs.cloudflare.com", "https://code.jquery.com", "https://kit.fontawesome.com", "https://www.google.com", "https://www.gstatic.com", "https://www.recaptcha.net"],
        scriptSrcAttr: ["'unsafe-inline'"],
        styleSrc: ["'self'", "'unsafe-inline'", "https://cdn.jsdelivr.net", "https://fonts.googleapis.com", "https://cdnjs.cloudflare.com"],
        styleSrcElem: ["'self'", "'unsafe-inline'", "https://cdn.jsdelivr.net", "https://fonts.googleapis.com", "https://cdnjs.cloudflare.com"],
        fontSrc: ["'self'", "https://fonts.gstatic.com", "https://cdn.jsdelivr.net", "https://cdnjs.cloudflare.com", "https://kit.fontawesome.com", "https://ka-f.fontawesome.com", "data:"],
        imgSrc: ["'self'", "data:", "blob:", "https:"],
        mediaSrc: ["'self'", "blob:"],
        connectSrc: ["'self'", "ws:", "wss:", "http://localhost:3000", "http://localhost:9000", "http://localhost:8000", "https://cdn.jsdelivr.net", "https://cdnjs.cloudflare.com", "https://www.google.com", "https://www.gstatic.com", "https://www.recaptcha.net", "https://ka-f.fontawesome.com"],
        frameSrc: ["'self'", "https://js.stripe.com", "https://www.google.com", "https://www.recaptcha.net"],
        objectSrc: ["'none'"],
        upgradeInsecureRequests: null,
      }),
    },
  };
}

module.exports = { createCookieSessionOptions, createHelmetOptions };
