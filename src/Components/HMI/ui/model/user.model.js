const mongoose = require('mongoose');

const UserSchema = new mongoose.Schema({
    username: {
        type: String,
        required: true,
        unique: true
    },
    email: {
        type: String,
        required: true,
        unique: true
    },
    password: {
        type: String,
        required: true
    },
    roles: [{
        type: String,
        enum: ['ROLE_USER', 'ROLE_ADMIN', 'ROLE_GUEST', 'user', 'admin', 'guest'],
        default: 'user'
    }],
    status: {
        type: String,
        enum: ['active', 'suspended', 'banned'],
        default: 'active'
    },
    blockMessage: String,
    createdAt: {
        type: Date,
        default: Date.now
    },
    lastLogin: Date
}, {
    timestamps: true
});

// Indexes
UserSchema.index({ email: 1 });
UserSchema.index({ status: 1 });
UserSchema.index({ createdAt: 1 });

// Export the model
const User = mongoose.model('User', UserSchema);
module.exports = User;