class HeaderNotifications {
    constructor() {
        this.notificationBell = document.getElementById('notificationBell');
        this.unreadCountBadge = document.getElementById('unreadCountBadge');
        this.unreadCountText = document.getElementById('unreadCountText');
        this.bellIcon = document.getElementById('bellIcon');
        this.socket = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.updateNotificationCount();
        this.setupRealTimeUpdates();
        setInterval(() => this.updateNotificationCount(), 30000);
    }

    setupEventListeners() {
        this.notificationBell.addEventListener('click', (e) => {
            e.preventDefault();
            window.location.href = '/admin-notifications';
        });
    }

    async updateNotificationCount() {
        try {
            const token = localStorage.getItem('token');
            if (!token) return;

            const response = await fetch('/api/notifications/unread-count', {
                headers: { 'Authorization': 'Bearer ' + token }
            });

            if (response.ok) {
                const data = await response.json();
                this.updateBadge(data.count || 0);
            }
        } catch (error) {
            console.error('Error fetching notification count:', error);
        }
    }

    updateBadge(unreadCount) {
        if (unreadCount > 0) {
            this.unreadCountBadge.style.display = 'flex';
            this.unreadCountText.textContent = unreadCount > 99 ? '99+' : unreadCount;

            // Add animation for new notifications
            this.unreadCountBadge.classList.add('notification-pulse');
            this.bellIcon.classList.add('bell-ring');

            setTimeout(() => {
                this.unreadCountBadge.classList.remove('notification-pulse');
                this.bellIcon.classList.remove('bell-ring');
            }, 1500);
        } else {
            this.unreadCountBadge.style.display = 'none';
        }
    }

    setupRealTimeUpdates() {
        if (typeof io !== 'undefined') {
            this.socket = io();

            this.socket.on('newNotification', () => {
                this.updateNotificationCount();
            });

            this.socket.on('notificationUpdated', () => {
                this.updateNotificationCount();
            });
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    new HeaderNotifications();
});