const { Notification } = require('../model/notification.model');
//const webpush = require('web-push');
const dispatchService = require('./dispatchService');

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
        this.typeConfig = {
            donation: { icon: 'ti ti-receipt-2', expiryDays: 30 },
            request: { icon: 'ti ti-alert-circle', expiryDays: 7 },
            user: { icon: 'ti ti-user', expiryDays: 14 },
            system: { icon: 'ti ti-bell', expiryDays: null },
            alert: { icon: 'ti ti-alert-triangle', expiryDays: 1 }
        };
    }

    async createNotification(userId, title, message, type = 'system', metadata = {}) {
        try {
            const typeConfig = this.typeConfig[type] || this.typeConfig.system;

            const notification = new Notification({
                userId,
                title,
                message,
                type,
                metadata,
                icon: typeConfig.icon,
                expiresAt: typeConfig.expiryDays ?
                    new Date(Date.now() + typeConfig.expiryDays * 24 * 60 * 60 * 1000) :
                    null
            });

            await notification.save();
            return notification;
        } catch (error) {
            console.error('Error creating notification:', error);
            throw error;
        }
    }

    async createAndDispatch(userId, title, message, type = 'system', metadata = {}, channel = null) {
        try {
            // Create the notification
            const notification = await this.createNotification(userId, title, message, type, metadata);

            // Dispatch it
            const dispatchResults = await dispatchService.dispatch(userId, notification, channel);

            return {
                notification,
                dispatchResults
            };
        } catch (error) {
            console.error('Error in createAndDispatch:', error);
            throw error;
        }
    }
    async getUserNotifications(userId, limit = 20, offset = 0) {
        try {
            return await Notification.find({ userId })
                .sort({ createdAt: -1 })
                .skip(offset)
                .limit(limit);
        } catch (error) {
            console.error('Error fetching user notifications:', error);
            throw error;
        }
    }

    async markAsRead(notificationId, userId) {
        try {
            return await Notification.findOneAndUpdate(
                { _id: notificationId, userId },
                { status: 'read' },
                { new: true }
            );
        } catch (error) {
            console.error('Error marking notification as read:', error);
            throw error;
        }
    }

    async deleteNotification(notificationId, userId) {
        try {
            return await Notification.findOneAndDelete({
                _id: notificationId,
                userId
            });
        }catch (error) {
            console.error('Error deleting notification:', error);
            throw error;
        }
    }

    getIconForType(type) {
        return this.typeConfig[type]?.icon || this.typeConfig.system.icon;
    }

    getExpiryForType(type) {
        const expiryDays = this.typeConfig[type]?.expiryDays || 7;
        return expiryDays ? new Date(Date.now() + expiryDays * 24 * 60 * 60 * 1000) : null;
    }
}

module.exports = new NotificationService();