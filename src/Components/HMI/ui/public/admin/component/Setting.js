const express = require('express');
const router = express.Router();
const { updateForce2FA, getForce2FA } = require('../controllers/settingsController');
const { requireAdminAuth } = require('../middleware/authMiddleware');

// Get current 2FA setting
router.get('/force2fa', requireAdminAuth, getForce2FA);

// Update 2FA setting
router.post('/force2fa', requireAdminAuth, updateForce2FA);

module.exports = router;
