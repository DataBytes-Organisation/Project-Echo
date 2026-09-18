const path = require('path');

function registerRoutes(app, dependencies) {
  const { publicDir, checkUserSession, redisClient, resolveLandingPath } = dependencies;

  const sendPage = (res, file) => res.sendFile(path.join(publicDir, file));

  const publicPages = [
    ['/login', 'pages/auth/login.html'],
    ['/login.html', 'pages/auth/login.html'],
    ['/verify-otp', 'pages/auth/verify-otp.html'],
    ['/verify-otp.html', 'pages/auth/verify-otp.html'],
    ['/forgotPassword', 'pages/auth/reset-password.html'],
    ['/resetPassword.html', 'pages/auth/reset-password.html'],
  ];

  const unprotectedAdminPages = [
    ['/admin/dashboard.html', 'pages/admin/dashboard.html'],
    ['/admin/admin-nodes.html', 'pages/admin/admin-nodes.html'],
    ['/admin/admin-nodes-temp.html', 'pages/admin/admin-nodes-temp.html'],
    ['/admin/cloud-compute.html', 'pages/admin/cloud-compute.html'],
    ['/admin/api-explorer.html', 'pages/admin/api-explorer.html'],
    ['/admin/projects.html', 'pages/admin/projects.html'],
    ['/admin/profile.html', 'pages/admin/profile.html'],
    ['/admin/template.html', 'pages/admin/template.html'],
    ['/admin/donations.html', 'pages/admin/donations.html'],
    ['/admin/admin-request.html', 'pages/admin/admin-request.html'],
    ['/admin/notifications.html', 'pages/admin/notifications.html'],
    ['/admin/sensor-health.html', 'pages/admin/sensor-health.html'],
    ['/admin/feedback.html', 'pages/admin/feedback.html'],
    ['/admin/hmi-data-insights.html', 'pages/admin/hmi-data-insights.html'],
    ['/admin/sensor_health/alerts.html', 'pages/admin/sensor_health/alerts.html'],
    ['/admin/sensor_health/reboot.html', 'pages/admin/sensor_health/reboot.html'],
    ['/admin/sensor_health/settings.html', 'pages/admin/sensor_health/settings.html'],
    ['/admin/sensor_health/add-project.html', 'pages/admin/sensor_health/add-project.html'],
    ['/admin/sensor_health/device-detail.html', 'pages/admin/sensor_health/device-detail.html'],
  ];

  for (const [route, file] of [...publicPages, ...unprotectedAdminPages]) {
    app.get(route, (req, res) => sendPage(res, file));
  }

  app.get(
    ['/admin*', '/map', '/requests', '/notifications'],
    checkUserSession
  );

  app.get('/', async (req, res) => {
    try {
      if (!redisClient.isOpen) await redisClient.connect();
      const storedToken = await redisClient.get('JWT');
      const role = await redisClient.get('Roles');
      return res.redirect(resolveLandingPath(req.session?.token, storedToken, role));
    } catch (error) {
      console.error('Landing redirect failed.');
      return res.redirect('/login');
    }
  });

  app.get('/admin-dashboard', (req, res) => {
    return sendPage(res, 'pages/admin/dashboard.html');
  });

  app.get('/admin-nodes', (req, res) => {
    return sendPage(res, 'pages/admin/admin-nodes.html');
  });

  app.get('/admin-nodes-temp', (req, res) => {
    return sendPage(res, 'pages/admin/admin-nodes-temp.html');
  });

  app.get('/admin-compute', (req, res) => {
    return sendPage(res, 'pages/admin/cloud-compute.html');
  });

  app.get('/admin-api-explorer', (req, res) => {
    return sendPage(res, 'pages/admin/api-explorer.html');
  });

  app.get('/admin-projects', (req, res) => {
    return sendPage(res, 'pages/admin/projects.html');
  });

  app.get('/admin-profile', (req, res) => {
    return sendPage(res, 'pages/admin/profile.html');
  });

  app.get('/admin-template', (req, res) => {
    return sendPage(res, 'pages/admin/template.html');
  });

  app.get('/admin-donations', (req, res) => {
    return sendPage(res, 'pages/admin/donations.html');
  });

  app.get('/requests', (req, res) => {
    sendPage(res, 'pages/admin/admin-request.html');
  });

  app.get('/notifications', (req, res) => {
    sendPage(res, 'pages/admin/notifications.html');
  });

  app.get('/welcome', async (req, res) => {
    try {
      if (!redisClient.isOpen) await redisClient.connect();
      const storedToken = await redisClient.get('JWT');
      const role = await redisClient.get('Roles');
      return res.redirect(resolveLandingPath(req.session?.token, storedToken, role));
    }
    catch {
      res.send('<script> alert("No user info detected! Please login again"); window.location.href = "/login"; </script>');
    }
  });

  app.get('/map', async (req, res) => {
    sendPage(res, 'pages/map/index.html');
  });

  app.get('/index.html', async (req, res) => {
    sendPage(res, 'pages/map/index.html');
  });
}

module.exports = { registerRoutes };
