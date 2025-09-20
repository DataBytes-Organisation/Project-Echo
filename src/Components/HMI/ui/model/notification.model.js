const mongoose = require('mongoose');

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
    archivedAt: Date,
    expiresAt: Date
}, {
    timestamps: true,
    index: { expiresAt: 1, expireAfterSeconds: 0 }
});

// Indexes
NotificationSchema.index({ userId: 1, status: 1 });
NotificationSchema.index({ userId: 1, archivedAt: -1 });
NotificationSchema.index({ userId: 1, createdAt: -1 });

const Notification = mongoose.model("Notification", NotificationSchema);

const UserNotificationPreferenceSchema = new mongoose.Schema({
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "User",
        required: true,
        unique: true
    },
    preferences: {
        channels: {
            inApp: { type: Boolean, default: true },
            email: { type: Boolean, default: false },
            push: { type: Boolean, default: false }
        },
        types: {
            system: { type: Boolean, default: true },
            donation: { type: Boolean, default: true },
            request: { type: Boolean, default: true },
            user: { type: Boolean, default: false }
        },
        doNotDisturb: {
            enabled: { type: Boolean, default: false },
            startTime: { type: String, default: '22:00' },
            endTime: { type: String, default: '07:00' },
            days: { type: [Number], default: [0, 1, 2, 3, 4, 5, 6] }
        }
    }
}, {
    timestamps: true,
    collection: 'UserNotificationPreference'
});

const UserNotificationPreference = mongoose.model('UserNotificationPreference', UserNotificationPreferenceSchema);

// Export both models
module.exports = {
    Notification,
    UserNotificationPreference
};