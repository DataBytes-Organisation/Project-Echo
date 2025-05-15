const express = require('express');
const router = express.Router();
const { getEnquiries, createEnquiry, updateEnquiry, deleteEnquiry } = require('../controller/enquiryController'); // Corrected path

// Route to get all enquiries
router.get('/', getEnquiries);

// Route to create a new enquiry
router.post('/', createEnquiry);

// Route to update an enquiry
router.patch('/:id', updateEnquiry);

// Route to delete an enquiry
router.delete('/:id', deleteEnquiry);

module.exports = router;
