const { UserNotificationPreference } = require('../model/user.model');
const axios = require('axios');

class DispatchService {
    constructor() {
        this.channels = {
            inApp: this.sendInApp,
            email: this.sendEmail,
            push: this.sendPush,
            sms: this.sendSms
        };
    }

    async dispatch(userId, notification, channel = null) {
        try {
            const preferences = await UserNotificationPreference.findOne({ userId }) ||
                new UserNotificationPreference({ userId });

            if (channel) {
                if (preferences.preferences[channel]) {
                    await this.channels[channel](userId, notification);
                }
            } else {
                for (const [channelName, isEnabled] of Object.entries(preferences.preferences)) {
                    if (isEnabled && this.channels[channelName]) {
                        await this.channels[channelName](userId, notification);
                    }
                }
            }
        } catch (error) {
            console.error(`Error dispatching notification to user ${userId}:`, error);
        }
    }

    async sendInApp(userId, notification) {
        // Handled by WebSocket
    }

    async sendEmail(userId, notification) {
        // Implement email sending
    }

    async sendPush(userId, notification) {
        // Implement push notification sending
    }

    async sendSms(userId, notification) {
        // Implement SMS sending
    }
}

module.exports = new DispatchService();