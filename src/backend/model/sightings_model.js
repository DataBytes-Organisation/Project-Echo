const mongoose = require('mongoose');

const SightingSchema = new mongoose.Schema({
  species: { type: String, required: true },
  latitude: { type: Number, required: true },
  longitude: { type: Number, required: true },
  datetime: { type: Date, required: true },
  notes: { type: String },
  filePath: { type: String },
});

module.exports = mongoose.model('Sighting', SightingSchema);
