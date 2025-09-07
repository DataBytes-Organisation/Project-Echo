const { UserNotificationPreference } = require('../model/notification.model');
const nodemailer = require('nodemailer');

class DispatchService {
    constructor() {
        this.channels = {
            inApp: this.sendInApp,
            email: this.sendEmail,
            push: this.sendPush
        };
        // Initialize email transporter (mock for now)
        this.transporter = nodemailer.createTransport({
            service: 'gmail',
            auth: {
                user: process.env.EMAIL_USER,
                pass: process.env.EMAIL_PASS
            }
        });
    }

    async dispatch(userId, notification, channel = null) {
        try {
            const preferences = await UserNotificationPreference.findOne({ userId }) ||
                new UserNotificationPreference({ userId });

            // Check Do Not Disturb first
            if (this.isInDoNotDisturb(preferences)) {
                console.log(`Notification suppressed for user ${userId} during DND hours`);
                return { suppressed: true, reason: 'DND' };
            }

            const results = {};

            if (channel) {
                // Send to specific channel only
                if (preferences.preferences[channel]) {
                    results[channel] = await this.channels[channel](userId, notification);
                }
            } else {
                // Send to all enabled channels
                for (const [channelName, isEnabled] of Object.entries(preferences.preferences)) {
                    if (isEnabled && this.channels[channelName]) {
                        results[channelName] = await this.channels[channelName](userId, notification);
                    }
                }
            }

            return results;
        } catch (error) {
            console.error(`Error dispatching notification to user ${userId}:`, error);
            throw error;
        }
    }

    async sendInApp(userId, notification) {
        try {
            // This would emit via Socket.io in a real implementation
            console.log(`In-app notification to user ${userId}: ${notification.title}`);

            // Simulate real-time notification
            if (global.io) {
                global.io.to(`user_${userId}`).emit('newNotification', notification);
            }

            return { success: true, channel: 'inApp' };
        } catch (error) {
            console.error('Error sending in-app notification:', error);
            return { success: false, error: error.message };
        }
    }

    async sendEmail(userId, notification) {
        try {
            // In a real implementation, you would:
            // 1. Look up user email
            // 2. Send actual email
            console.log(`Email notification to user ${userId}: ${notification.title}`);

            // Mock implementation
            const mailOptions = {
                from: process.env.EMAIL_FROM,
                to: 'user@example.com', // Would be user's actual email
                subject: notification.title,
                text: notification.message,
                html: this.generateEmailTemplate(notification)
            };

            await this.transporter.sendMail(mailOptions);
            return { success: true, channel: 'email' };
        } catch (error) {
            console.error('Error sending email notification:', error);
            return { success: false, error: error.message };
        }
    }

    async sendPush(userId, notification) {
        try {
            console.log(`Push notification to user ${userId}: ${notification.title}`);
            // Push notification logic would go here
            return { success: true, channel: 'push' };
        } catch (error) {
            console.error('Error sending push notification:', error);
            return { success: false, error: error.message };
        }
    }

    isInDoNotDisturb(preferences) {
        if (!preferences.doNotDisturb?.enabled) return false;

        const now = new Date();
        const [startHour, startMinute] = preferences.doNotDisturb.startTime.split(':').map(Number);
        const [endHour, endMinute] = preferences.doNotDisturb.endTime.split(':').map(Number);

        const startTime = new Date();
        startTime.setHours(startHour, startMinute, 0, 0);

        const endTime = new Date();
        endTime.setHours(endHour, endMinute, 0, 0);

        if (startTime > endTime) {
            return now >= startTime || now <= endTime;
        }

        return now >= startTime && now <= endTime;
    }

    generateEmailTemplate(notification) {
        return `
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <h2>${notification.title}</h2>
                <p>${notification.message}</p>
                <hr>
                <p style="color: #666; font-size: 12px;">
                    Sent from Notification System • ${new Date().toLocaleString()}
                </p>
            </div>
        `;
    }
}

module.exports = new DispatchService();