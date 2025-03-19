const SightingsModel = require('../model/sightings_model');

exports.processSighting = async (req, res) => {
  try {
    const { species, latitude, longitude, datetime, notes } = req.body;
    const filePath = req.file ? req.file.path : null;

    const newSighting = new SightingsModel({
      species,
      latitude,
      longitude,
      datetime,
      notes,
      filePath,
    });

    await newSighting.save();
    res.status(201).send({ message: 'Sighting uploaded successfully!' });
  } catch (error) {
    console.error('Error processing sighting:', error);
    res.status(500).send({ error: 'Failed to upload sighting.' });
  }
};
