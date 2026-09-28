function registerRoutes(app, dependencies) {
  const { requireApiSession, notificationStore } = dependencies;

  app.get('/api/notifications', requireApiSession, async (req, res) => {
    try {
      res.json(await notificationStore.listNotifications());
    } catch (error) {
      console.error('Error loading notifications:', error);
      res.status(500).json({ error: 'Unable to load notifications.' });
    }
  });

  // Specific actions must stay before /:id routes.
  app.patch('/api/notifications/read-all', requireApiSession, async (req, res) => {
    try {
      const { notifications } = await notificationStore.listNotifications();
      const unreadIds = notifications.filter(item => !item.read).map(item => item.id);
      await notificationStore.setNotificationFlags(unreadIds, { read: true });
      res.json(await notificationStore.listNotifications());
    } catch (error) {
      console.error('Error marking all notifications read:', error);
      res.status(500).json({ error: 'Unable to mark notifications as read.' });
    }
  });

  app.delete('/api/notifications/read', requireApiSession, async (req, res) => {
    try {
      const { notifications } = await notificationStore.listNotifications();
      const readIds = notifications.filter(item => item.read).map(item => item.id);
      await notificationStore.setNotificationFlags(readIds, { deleted: true });
      res.json(await notificationStore.listNotifications());
    } catch (error) {
      console.error('Error deleting read notifications:', error);
      res.status(500).json({ error: 'Unable to delete read notifications.' });
    }
  });

  app.patch('/api/notifications/:id/read', requireApiSession, async (req, res) => {
    try {
      await notificationStore.setNotificationFlags([req.params.id], { read: true });
      res.json(await notificationStore.listNotifications());
    } catch (error) {
      console.error('Error marking notification read:', error);
      res.status(500).json({ error: 'Unable to mark the notification as read.' });
    }
  });

  app.delete('/api/notifications/:id', requireApiSession, async (req, res) => {
    try {
      await notificationStore.setNotificationFlags([req.params.id], { deleted: true });
      res.json(await notificationStore.listNotifications());
    } catch (error) {
      console.error('Error deleting notification:', error);
      res.status(500).json({ error: 'Unable to delete the notification.' });
    }
  });
}

module.exports = { registerRoutes };
