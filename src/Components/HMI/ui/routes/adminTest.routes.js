const express = require('express');
const router = express.Router();
const User = require('../model/user.model');
const bcrypt = require('bcryptjs');
const crypto = require('crypto');

// Generate random password
const generateRandomPassword = () => crypto.randomBytes(6).toString('hexx');

// ✅ Route: Reset password (test-only)
router.put('/dev/users/:id/reset-password', async (req, res) => {
  try {
    const newPassword = generateRandomPassword();
    const hashedPassword = await bcrypt.hash(newPassword, 8);

    const user = await User.findByIdAndUpdate(
      req.params.id,
      { password: hashedPassword },
      { new: true }
    );

    if (!user) return res.status(404).json({ error: 'User not found' });

    console.log(`[TEST] Reset password for ${user.email} → ${newPassword}`);
    res.json({ message: 'Password reset', newPassword }); // for dev only
  } catch (err) {
    console.error('[TEST] Error resetting password:', err);
    res.status(500).json({ error: 'Failed to reset password' });
  }
});

// ✅ Route: Toggle MFA (test-only)
router.put('/dev/users/:userId/toggle-mfa', async (req, res) => {
  try {
    const user = await User.findById(req.params.userId);
    if (!user) return res.status(404).json({ error: 'User not found' });

    user.mfaEnabled = !user.mfaEnabled;
    await user.save();

    console.log(`[TEST] Toggled MFA for ${user.email} → ${user.mfaEnabled}`);
    res.status(200).json({ message: 'MFA toggled', mfaEnabled: user.mfaEnabled });
  } catch (err) {
    console.error('[TEST] Error toggling MFA:', err);
    res.status(500).json({ error: 'Failed to toggle MFA' });
  }
});

module.exports = router;