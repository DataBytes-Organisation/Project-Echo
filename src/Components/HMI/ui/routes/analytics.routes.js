const express = require('express');
const router = express.Router();
const User = require('../model/user.model');

// DEBUG: confirm router is loaded
console.log('✅ analytics.routes.js loaded');

// TEST ROUTE
router.get('/test', (req, res) => {
  console.log('✅ /api/test hit');
  res.json({ ok: true });
});

router.post('/track-visit', async (req, res) => {
    console.log('Hit /api/track-visit route');
  
    try {
      const { email, username } = req.body;
  
      if (!email && !username) {
        return res.status(400).json({ error: 'Email or username is required' });
      }
  
      const user = await User.findOne(email ? { email } : { username });
  
      if (!user) {
        return res.status(404).json({ error: 'User not found' });
      }
  
      user.set('visitCount', (user.visitCount || 0) + 1);
      user.set('lastVisit', new Date());
  
      console.log(`[DEBUG] Saving visit for ${user.email}`);
      console.log(`[DEBUG] visitCount: ${user.visitCount}, lastVisit: ${user.lastVisit}`);
  
      const saved = await user.save();
  
      console.log(`[DEBUG] Save result:`, saved);
  
      res.status(200).json({ message: 'Visit recorded' });
    } catch (err) {
      console.error('Error tracking visit:', err);
      res.status(500).json({ error: 'Failed to track visit' });
    }
});

module.exports = router;