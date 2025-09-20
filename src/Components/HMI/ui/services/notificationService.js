const { Notification, UserNotificationPreference } = require('../model/notification.model');
const dispatchService = require('./dispatchService');

class NotificationService {
    constructor() {
        console.log('NotificationService initialized');
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
            const preferences = await UserNotificationPreference.findOne({ userId }) ||
                new UserNotificationPreference({ userId });

            // Check if this notification type is enabled
            if (!this.isNotificationTypeEnabled(preferences, type)) {
                console.log(`Notification type ${type} is disabled for user ${userId}`);
                return {
                    notification: null,
                    dispatchResults: { suppressed: true, reason: 'type_disabled' }
                };
            }

            // Check Do Not Disturb
            if (type !== 'alert' && this.isInDoNotDisturb(preferences)) {
                console.log(`Notification suppressed for user ${userId} during DND hours`);
                return {
                    notification: null,
                    dispatchResults: { suppressed: true, reason: 'DND' }
                };
            }

            // Only create notification if it will be dispatched
            const notification = await this.createNotification(userId, title, message, type, metadata);
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

    isNotificationTypeEnabled(preferences, notificationType) {
        // Critical alerts always go through
        if (notificationType === 'alert') {
            return true;
        }

        // Check if this specific type is enabled in user preferences
        const typeEnabled = preferences.preferences.types[notificationType];

        if (typeEnabled !== undefined) {
            return typeEnabled;
        }

        // Fallback to default configuration
        return this.typeConfig[notificationType]?.defaultEnabled || false;
    }

    isInDoNotDisturb(preferences) {
        if (!preferences.preferences.doNotDisturb?.enabled) {
            return false;
        }

        const now = new Date();
        const currentDay = now.getDay();

        if (!preferences.preferences.doNotDisturb.days.includes(currentDay)) {
            return false;
        }

        const [startHour, startMinute] = preferences.preferences.doNotDisturb.startTime.split(':').map(Number);
        const [endHour, endMinute] = preferences.preferences.doNotDisturb.endTime.split(':').map(Number);

        const startTime = new Date();
        startTime.setHours(startHour, startMinute, 0, 0);

        const endTime = new Date();
        endTime.setHours(endHour, endMinute, 0, 0);

        if (startTime > endTime) {
            return now >= startTime || now <= endTime;
        }

        return now >= startTime && now <= endTime;
    }

    async getUserNotifications(userId, limit = 20, offset = 0, status = null) {
        try {
            const query = { userId };
            if (status) {
                query.status = status;
            }

            return await Notification.find(query)
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

    async markAllAsRead(userId) {
        try {
            return await Notification.updateMany(
                { userId: userId, status: 'unread'},
                { status: 'read' }
            );
        } catch (error) {
            console.error('Error marking all notifications as read:', error);
            throw error;
        }
    }

    async deleteNotification(notificationId, userId) {
        try {
            return await Notification.findOneAndDelete({
                _id: notificationId,
                userId
            });
        } catch (error) {
            console.error('Error deleting notification:', error);
            throw error;
        }
    }

    async getUnreadCount(userId) {
        try {
            return await Notification.countDocuments({
                userId: userId,
                status: 'unread'
            });
        } catch (error) {
            console.error('Error getting unread count:', error);
            throw error;
        }
    }

    async archiveNotification(notificationId, userId) {
        try {
            const notification = await Notification.findOneAndUpdate(
                {
                    _id: notificationId,
                    userId: userId
                },
                {
                    status: 'archived',
                    archivedAt: new Date()
                },
                { new: true }
            );

            if (!notification) {
                throw new Error('Notification not found or access denied');
            }

            return notification;
        } catch (error) {
            console.error('Error archiving notification:', error);
            throw error;
        }
    }

    async archiveAllNotifications(userId) {
        try {
            const result = await Notification.updateMany(
                {
                    userId: userId,
                    status: { $in: ['unread', 'read'] }
                },
                {
                    status: 'archived',
                    archivedAt: new Date()
                }
            );

            return result;
        } catch (error) {
            console.error('Error archiving all notifications:', error);
            throw error;
        }
    }

    async getArchivedNotifications(userId, limit = 20, offset = 0) {
        try {
            return await Notification.find({
                userId: userId,
                status: 'archived'
            })
                .sort({ archivedAt: -1, createdAt: -1 })
                .skip(offset)
                .limit(limit);
        } catch (error) {
            console.error('Error fetching archived notifications:', error);
            throw error;
        }
    }

    async restoreNotification(notificationId, userId) {
        try {
            const notification = await Notification.findOneAndUpdate(
                {
                    _id: notificationId,
                    userId: userId,
                    status: 'archived'
                },
                {
                    status: 'read',
                    archivedAt: null
                },
                { new: true }
            );

            if (!notification) {
                throw new Error('Archived notification not found or access denied');
            }

            return notification;
        } catch (error) {
            console.error('Error restoring notification:', error);
            throw error;
        }
    }

    async getUserNotificationsWithFilter(userId, limit = 20, offset = 0, status = null) {
        try {
            // Get user preferences
            const preferences = await UserNotificationPreference.findOne({ userId }) ||
                new UserNotificationPreference({ userId });

            const query = { userId };
            if (status) {
                query.status = status;
            }

            // Get all notifications first
            let notifications = await Notification.find(query)
                .sort({ createdAt: -1 })
                .skip(offset)
                .limit(limit);

            // Filter out notifications where the type is disabled in user preferences
            notifications = notifications.filter(notification => {
                const typeEnabled = preferences.preferences.types[notification.type];
                return typeEnabled !== false; // Only show if explicitly enabled or undefined
            });

            return notifications;
        } catch (error) {
            console.error('Error fetching user notifications with filter:', error);
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