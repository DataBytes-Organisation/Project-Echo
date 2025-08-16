const mongoose = require('mongoose');
const donationSchema = new mongoose.Schema({
    amount: {
        type: Number,
        required: true
    },
    status: {
        type: String,
        enum: ['succeeded', 'pending', 'failed'],
        required: true
    },
    billing_details: {
        email: {
            type: String,
            required: true
        }
    },
    created: {
        type: Date,
        default: Date.now
    },
    type: {
        type: String,
        enum: ['One-Time', 'Recurring'],
        required: true
    }
}, {
    timestamps: true
});

module.exports = mongoose.model('Donation', donationSchema);
