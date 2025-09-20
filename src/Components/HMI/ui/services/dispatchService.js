const { UserNotificationPreference } = require('../model/notification.model');
const { User } = require('../model/user.model');
const nodemailer = require('nodemailer');

class DispatchService {
    constructor() {
        this.channels = {
            inApp: this.sendInApp.bind(this),
            email: this.sendEmail.bind(this),
            push: this.sendPush.bind(this)
        };

        // Notification type configuration
        this.typeConfig = {
            system: { defaultEnabled: true, priority: 'high' },
            donation: { defaultEnabled: true, priority: 'medium' },
            request: { defaultEnabled: true, priority: 'high' },
            user: { defaultEnabled: false, priority: 'low' },
            alert: { defaultEnabled: true, priority: 'critical' }
        };

        // Initialize email transporter
        this.transporter = nodemailer.createTransport({
            service: 'gmail',
            auth: {
                user: process.env.EMAIL_USER || 'echodatabytes@gmail.com',
                pass: process.env.EMAIL_PASS || 'ltzoycrrkpeipngi'
            }
        });
    }

    async dispatch(userId, notification, channel = null) {
        try {
            const preferences = await UserNotificationPreference.findOne({ userId }) ||
                new UserNotificationPreference({ userId });

            console.log('User preferences:', JSON.stringify(preferences, null, 2));

            // Initialize results object
            const results = {};

            if (channel) {
                // Send to specific channel only if enabled
                if (preferences.preferences.channels[channel]) {
                    results[channel] = await this.channels[channel](userId, notification, preferences);
                }
            } else {
                // Send to all enabled channels
                for (const [channelName, isEnabled] of Object.entries(preferences.preferences.channels)) {
                    if (isEnabled && this.channels[channelName]) {
                        results[channelName] = await this.channels[channelName](userId, notification, preferences);
                    }
                }
            }

            return results;
        } catch (error) {
            console.error(`Error dispatching notification to user ${userId}:`, error);
            throw error;
        }
    }

    async sendInApp(userId, notification, preferences) {
        try {
            console.log(`In-app notification to user ${userId}: ${notification.title}`);

            // Emit via Socket.io
            if (global.io) {
                global.io.to(`user_${userId}`).emit('newNotification', {
                    ...notification.toObject ? notification.toObject() : notification,
                    timestamp: new Date()
                });
            }

            return { success: true, channel: 'inApp' };
        } catch (error) {
            console.error('Error sending in-app notification:', error);
            return { success: false, error: error.message };
        }
    }

    async sendEmail(userId, notification, preferences) {
        try {
            // Get user email from database
            const user = await User.findById(userId);
            if (!user || !user.email) {
                console.log(`User ${userId} not found or no email address`);
                return { success: false, error: 'User email not found' };
            }

            console.log(`Email notification to user ${userId} (${user.email}): ${notification.title}`);

            const mailOptions = {
                from: process.env.EMAIL_FROM || 'echodatabytes@gmail.com',
                to: user.email,
                subject: `EchoNet: ${notification.title}`,
                text: this.generateEmailText(notification),
                html: this.generateEmailTemplate(notification)
            };

            await this.transporter.sendMail(mailOptions);
            return { success: true, channel: 'email' };
        } catch (error) {
            console.error('Error sending email notification:', error);
            return { success: false, error: error.message };
        }
    }

    async sendPush(userId, notification, preferences) {
        try {
            console.log(`Push notification to user ${userId}: ${notification.title}`);

            // TODO: Implement actual push notification logic
            // This would integrate with Firebase Cloud Messaging, OneSignal, etc.

            return { success: true, channel: 'push' };
        } catch (error) {
            console.error('Error sending push notification:', error);
            return { success: false, error: error.message };
        }
    }

    generateEmailText(notification) {
        return `
${notification.title}

${notification.message}

${notification.link ? `View more: ${notification.link}` : ''}

---
Sent from EchoNet Notification System
${new Date().toLocaleString()}
        `.trim();
    }

    generateEmailTemplate(notification) {
        const iconConfig = {
            system: '🔔',
            donation: '💰',
            request: '📝',
            user: '👤',
            alert: '🚨'
        };

        const icon = iconConfig[notification.type] || '🔔';

        return `
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <style>
                    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                    .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                    .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; border-radius: 10px 10px 0 0; }
                    .content { background: #f9f9f9; padding: 20px; border-radius: 0 0 10px 10px; }
                    .notification-icon { font-size: 48px; margin-bottom: 20px; }
                    .button { display: inline-block; padding: 12px 24px; background: #667eea; color: white; text-decoration: none; border-radius: 5px; margin-top: 20px; }
                    .footer { margin-top: 20px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <div class="notification-icon">${icon}</div>
                        <h1>EchoNet Notification</h1>
                    </div>
                    <div class="content">
                        <h2>${notification.title}</h2>
                        <p>${notification.message}</p>
                        
                        ${notification.link ? `
                        <div style="text-align: center; margin: 30px 0;">
                            <a href="${notification.link}" class="button">View Details</a>
                        </div>
                        ` : ''}
                        
                        <div class="footer">
                            <p>Sent from EchoNet Notification System • ${new Date().toLocaleString()}</p>
                            <p><small>You can manage your notification preferences in your account settings.</small></p>
                        </div>
                    </div>
                </div>
            </body>
            </html>
        `;
    }

    // Helper method to check if user should receive a specific type of notification
    async shouldReceiveNotification(userId, notificationType) {
        try {
            const preferences = await UserNotificationPreference.findOne({ userId }) ||
                new UserNotificationPreference({ userId });

            return this.isNotificationTypeEnabled(preferences, notificationType) &&
                !this.isInDoNotDisturb(preferences);
        } catch (error) {
            console.error('Error checking notification preferences:', error);
            return true; // Default to allowing notifications on error
        }
    }

    // Method to get user notification settings
    async getUserNotificationSettings(userId) {
        try {
            const preferences = await UserNotificationPreference.findOne({ userId }) ||
                new UserNotificationPreference({ userId });

            return {
                channels: preferences.preferences.channels,
                types: preferences.preferences.types,
                doNotDisturb: preferences.preferences.doNotDisturb
            };
        } catch (error) {
            console.error('Error getting user notification settings:', error);
            return null;
        }
    }
}

module.exports = new DispatchService();