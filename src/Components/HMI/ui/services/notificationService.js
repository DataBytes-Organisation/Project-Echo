const { Notification } = require('../model/notification.model');
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

    getIconForType(type) {
        return this.typeConfig[type]?.icon || this.typeConfig.system.icon;
    }

    getExpiryForType(type) {
        const expiryDays = this.typeConfig[type]?.expiryDays || 7;
        return expiryDays ? new Date(Date.now() + expiryDays * 24 * 60 * 60 * 1000) : null;
    }
}

module.exports = new NotificationService();