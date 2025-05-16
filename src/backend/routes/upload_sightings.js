const express = require('express');
const multer = require('multer');
const { processSighting } = require('../controllers/upload_sightings_controller.js');

const router = express.Router();

const upload = multer({ dest: 'assets/uploads/' });

router.post('/', upload.single('file'), processSighting);

module.exports = router;
