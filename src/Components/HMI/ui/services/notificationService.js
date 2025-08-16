const { Notification, UserNotificationPreference } = require('../model/user.model');
const axios = require('axios');
const webpush = require('web-push');

class NotificationService {
    constructor() {
        console.log('NotificationService initialized (push notifications disabled)');
        // Initialize web push (for browser notifications)
        /**
        webpush.setVapidDetails(
            `mailto:${process.env.VAPID_EMAIL}`,
            process.env.VAPID_PUBLIC_KEY,
            process.env.VAPID_PRIVATE_KEY
        );**/
    }

    async createNotification(userId, title, message, type, metadata = {}) {
        try {
            const notification = new Notification({
                userId,
                title,
                message,
                type,
                metadata,
                icon: this.getIconForType(type),
                expiresAt: this.getExpiryForType(type)
            });

            await notification.save();

            // Dispatch to all channels based on user preferences
            await this.dispatchNotification(notification);

            return notification;
        } catch (error) {
            console.error('Error creating notification:', error);
            throw error;
        }
    }

    async dispatchNotification(notification) {
        try {
            const preferences = await UserNotificationPreference.findOne({ userId: notification.userId }) ||
                new UserNotificationPreference({ userId: notification.userId });

            // Check if in do-not-disturb window
            if (this.isInDoNotDisturb(preferences)) {
                console.log(`Notification suppressed for user ${notification.userId} during DND hours`);
                return;
            }

            // In-app notification (always sent as it's stored in DB)
            if (preferences.preferences.inApp) {
                this.sendInAppNotification(notification);
            }

            // Email notification
            if (preferences.preferences.email) {
                this.sendEmailNotification(notification);
            }

            // Push notification
            if (preferences.preferences.push) {
                this.sendPushNotification(notification);
            }

            // SMS notification (would require SMS service integration)
            if (preferences.preferences.sms) {
                this.sendSmsNotification(notification);
            }
        } catch (error) {
            console.error('Error dispatching notification:', error);
        }
    }

    sendInAppNotification(notification) {
        // This will be handled by the WebSocket connection
        console.log(`In-app notification sent for ${notification.userId}`);
    }

    async sendEmailNotification(notification) {
        // Implement email sending logic using your email service
        console.log(`Email notification sent for ${notification.userId}`);
    }

    async sendPushNotification(notification) {
        try {
            // In a real app, you would look up the user's subscription
            // const subscription = await getPushSubscription(notification.userId);
            // await webpush.sendNotification(subscription, JSON.stringify({
            //   title: notification.title,
            //   body: notification.message,
            //   icon: notification.icon,
            //   data: { url: notification.link }
            // }));
            console.log(`Push notification sent for ${notification.userId}`);
        } catch (error) {
            console.error('Error sending push notification:', error);
        }
    }

    sendSmsNotification(notification) {
        // Implement SMS sending logic
        console.log(`SMS notification sent for ${notification.userId}`);
    }

    getIconForType(type) {
        const icons = {
            donation: 'ti ti-receipt-2',
            request: 'ti ti-alert-circle',
            user: 'ti ti-user',
            system: 'ti ti-bell',
            alert: 'ti ti-alert-triangle'
        };
        return icons[type] || 'ti ti-bell';
    }

    getExpiryForType(type) {
        const expiryDays = {
            donation: 30,    // 30 days for donations
            request: 7,      // 1 week for requests
            user: 14,        // 2 weeks for user notifications
            system: null,    // Never expire for system notifications
            alert: 1         // 1 day for alerts
        };

        const days = expiryDays[type] || 7;
        return days ? new Date(Date.now() + days * 24 * 60 * 60 * 1000) : null;
    }

    isInDoNotDisturb(preferences) {
        if (!preferences.doNotDisturb.enabled) return false;

        const now = new Date();
        const [startHour, startMinute] = preferences.doNotDisturb.startTime.split(':').map(Number);
        const [endHour, endMinute] = preferences.doNotDisturb.endTime.split(':').map(Number);

        const startTime = new Date();
        startTime.setHours(startHour, startMinute, 0, 0);

        const endTime = new Date();
        endTime.setHours(endHour, endMinute, 0, 0);

        // Handle overnight DND (e.g., 22:00 to 08:00)
        if (startTime > endTime) {
            return now >= startTime || now <= endTime;
        }

        return now >= startTime && now <= endTime;
    }
}

module.exports = new NotificationService();