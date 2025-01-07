const { Metrics, Leaderboard } = require('../model/uex_model');

const getMetrics = async (req, res) => {
    try {
        const metrics = await Metrics.findOne(); 
        if (!metrics) {
            return res.status(404).json({ message: 'Metrics not found' });
        }
        res.json(metrics);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
};


const getLeaderboard = async (req, res) => {
    try {
        const leaderboard = await Leaderboard.find().sort({ uploads: -1 }).limit(10); 
        res.json(leaderboard);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
};

module.exports = { getMetrics, getLeaderboard };

