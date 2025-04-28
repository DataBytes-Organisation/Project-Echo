const Settings = require('../models/Settings');

// GET current 2FA setting
const getForce2FA = async (req, res) => {
  try {
    const settings = await Settings.findOne();
    res.json({ force2fa: settings?.force2fa || false });
  } catch (err) {
    res.status(500).json({ error: 'Failed to fetch setting' });
  }
};

// POST to update 2FA setting
const updateForce2FA = async (req, res) => {
  try {
    const { force2fa } = req.body;
    let settings = await Settings.findOne();

    if (!settings) {
      settings = new Settings({ force2fa });
    } else {
      settings.force2fa = force2fa;
    }

    await settings.save();
    res.status(200).json({ message: 'Force 2FA updated successfully.' });
  } catch (err) {
    res.status(500).json({ error: 'Failed to update setting' });
  }
};

module.exports = {
  getForce2FA,
  updateForce2FA,
};
