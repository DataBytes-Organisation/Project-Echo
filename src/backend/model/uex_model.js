const mongoose = require('mongoose');

const MetricsSchema = new mongoose.Schema({
    totalSightings: { type: Number, required: true },
    avgUploadTime: { type: String, required: true },
    satisfactionScore: { type: Number, required: true }
});

const LeaderboardSchema = new mongoose.Schema({
    name: { type: String, required: true },
    uploads: { type: Number, required: true }
});

const Metrics = mongoose.model('Metrics', MetricsSchema);
const Leaderboard = mongoose.model('Leaderboard', LeaderboardSchema);

module.exports = { Metrics, Leaderboard };

