const mongoose = require("mongoose");

const NotificationSchema = new mongoose.Schema({
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "User",
        required: true
    },
    message: {
        type: String,
        required: true
    },
    type: {
        type: String,
        enum: ['donation', 'request', 'user', 'system'],
        required: true
    },
    read: {
        type: Boolean,
        default: false
    },
    link: String,
    icon: String,
    metadata: mongoose.Schema.Types.Mixed // For any additional data
}, {
    timestamps: true,
    toJSON: { virtuals: true },
    toObject: { virtuals: true }
});

module.exports = mongoose.model("Notification", NotificationSchema);