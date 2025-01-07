const express = require('express');
const router = express.Router();
const { getMetrics, getLeaderboard } = require('../controllers/uex_metrics_controller');

router.get('/metrics', getMetrics);
router.get('/leaderboard', getLeaderboard);

module.exports = router;
