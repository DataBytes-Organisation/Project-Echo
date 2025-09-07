const mongoose = require("mongoose");

const NotificationSchema = new mongoose.Schema({
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "User",
        required: true
    },
    title: {
        type: String,
        required: true
    },
    message: {
        type: String,
        required: true
    },
    type: {
        type: String,
        enum: ['donation', 'request', 'user', 'system', 'alert'],
        required: true
    },
    status: {
        type: String,
        enum: ['unread', 'read', 'archived'],
        default: 'unread'
    },
    link: String,
    icon: String,
    metadata: mongoose.Schema.Types.Mixed,
    expiresAt: Date
}, {
    timestamps: true,
    index: { expiresAt: 1, expireAfterSeconds: 0 } // For auto-deleting expired notifications
});

const UserNotificationPreferenceSchema = new mongoose.Schema({
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "User",
        required: true,
        unique: true
    },
    preferences: {
        inApp: { type: Boolean, default: true },
        email: { type: Boolean, default: false },
        push: { type: Boolean, default: false }
    },
    doNotDisturb: {
        enabled: { type: Boolean, default: false },
        startTime: String, // "22:00"
        endTime: String    // "08:00"
    }
});

const Notification = mongoose.model("Notification", NotificationSchema);
const UserNotificationPreference = mongoose.model("UserNotificationPreference", UserNotificationPreferenceSchema);

module.exports = { Notification, UserNotificationPreference };