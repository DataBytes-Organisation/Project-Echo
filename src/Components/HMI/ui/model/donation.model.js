const mongoose = require('mongoose');

const DonationSchema = new mongoose.Schema({
    amount: {
        type: Number,
        required: true
    },
    currency: {
        type: String,
        default: 'aud'
    },
    donorEmail: {
        type: String,
        required: true
    },
    stripeChargeId: {
        type: String,
        required: true,
        unique: true
    },
    status: {
        type: String,
        enum: ['succeeded', 'pending', 'failed'],
        default: 'succeeded'
    },
    receiptUrl: String,
    createdAt: {
        type: Date,
        default: Date.now
    }
});

// Export the model
const Donation = mongoose.model('Donation', DonationSchema);
module.exports = Donation;