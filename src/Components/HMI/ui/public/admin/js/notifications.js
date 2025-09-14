$(document).ready(function() {
    // Verify Socket.IO is available
    if (typeof io === 'undefined') {
        console.error('Socket.IO not loaded!');
        return;
    }

    const token = localStorage.getItem('token');
    if (!token) {
        console.error('No token found in localStorage');
        window.location.href = '/login';
        return;
    }

    // Connect to Socket.io
    const socket = io('http://localhost:8080', {
        auth: { token },
        reconnection: true,
        reconnectionAttempts: Infinity,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        query: { token }
    });

    // Connection status logging
    socket.on('connect', () => {
        console.log('Socket.IO connected with ID:', socket.id);
        updateConnectionStatus(true);
    });

    socket.on('disconnect', (reason) => {
        console.log('Socket.IO disconnected:', reason);
        updateConnectionStatus(false);
        if (reason === 'io server disconnect') {
            socket.connect();
        }
    });

    socket.on('connect_error', (err) => {
        console.error('Socket.IO connection error:', err.message);
        if (err.message.includes('auth')) {
            socket.auth.token = localStorage.getItem('token');
            socket.connect();
        }
    });

    let notifications = [];
    let unreadCount = 0;
    let currentTab = 'all'; // 'all', 'unread', 'archived'

    // DOM Elements
    const notificationList = $("#notification-list");
    const unreadBadge = $("#unread-badge");
    const markAllReadBtn = $("#mark-all-read");
    const loadMoreBtn = $("#load-more");
    const archiveAllBtn = $("#archive-all");
    const connectionIndicator = $("#connection-indicator");

    // Tab elements
    const allTab = $("#all-tab");
    const unreadTab = $("#unread-tab");
    const archivedTab = $("#archived-tab");

    let isLoading = false;
    let offset = 0;
    const limit = 10;

    // Initialize
    loadNotifications();
    setupSocketListeners();
    setupEventHandlers();

    function updateConnectionStatus(connected) {
        connectionIndicator
            .removeClass(connected ? 'disconnected' : 'connected')
            .addClass(connected ? 'connected' : 'disconnected')
            .text(connected ? 'Connected' : 'Disconnected');
    }

    function loadNotifications(status = null) {
        if (isLoading) return;

        isLoading = true;
        loadMoreBtn.prop('disabled', true).html('<i class="fas fa-spinner fa-spin"></i> Loading...');

        let url = `/api/notifications?limit=${limit}&offset=${offset}`;
        if (status === 'archived') {
            url = `/api/notifications/archived?limit=${limit}&offset=${offset}`;
        } else if (status) {
            url += `&status=${status}`;
        }

        $.ajax({
            url: url,
            headers: {
                'Authorization': 'Bearer ' + token,
                'Content-Type': 'application/json'
            },
            success: function(response) {
                if (offset === 0) {
                    notifications = response.notifications || [];
                    renderNotifications();
                } else {
                    notifications = notifications.concat(response.notifications || []);
                    appendNotifications(response.notifications || []);
                }
                offset += (response.notifications || []).length;
                updateUnreadCount();
            },
            error: function(xhr) {
                console.error('API Error:', xhr.responseText);
                showErrorToast('Failed to load notifications');
            },
            complete: function() {
                isLoading = false;
                loadMoreBtn.prop('disabled', false).html('Load More');

                if (notifications.length > 0 && notifications.length % limit !== 0) {
                    loadMoreBtn.hide();
                } else {
                    loadMoreBtn.show();
                }
            }
        });
    }

    function showErrorToast(message) {
        Toastify({
            text: message,
            duration: 5000,
            gravity: "top",
            position: "right",
            backgroundColor: "#dc3545",
            stopOnFocus: true
        }).showToast();
    }

    function showSuccessToast(message) {
        Toastify({
            text: message,
            duration: 5000,
            gravity: "top",
            position: "right",
            backgroundColor: "#28a745",
            stopOnFocus: true
        }).showToast();
    }

    function renderNotifications() {
        notificationList.empty();

        if (notifications.length === 0) {
            notificationList.html(`
                <div class="empty-notifications">
                    <i class="far fa-bell-slash"></i>
                    <h5>No notifications yet</h5>
                    <p>You're all caught up!</p>
                </div>
            `);
            return;
        }

        notifications.forEach(notification => {
            notificationList.append(createNotificationElement(notification));
        });
    }

    function appendNotifications(newNotifications) {
        newNotifications.forEach(notification => {
            notificationList.append(createNotificationElement(notification));
        });
    }

    function createNotificationElement(notification) {
        const timeAgo = moment(notification.createdAt).fromNow();
        const isArchived = notification.status === 'archived';

        return `
            <div class="notification-item ${notification.status}" data-id="${notification._id}">
                <div class="notification-icon">
                    <i class="${notification.icon || 'ti ti-bell'}"></i>
                </div>
                <div class="notification-content">
                    <h5 class="notification-title">${notification.title}</h5>
                    <p class="notification-message">${notification.message}</p>
                    <div class="notification-footer">
                        <small class="notification-time">${timeAgo}</small>
                        <div class="notification-actions">
                            ${notification.link ?
            `<a href="${notification.link}" class="btn btn-sm btn-link">View</a>` : ''}
                            
                            ${!isArchived ? `
                                <button class="btn btn-sm ${notification.status === 'unread' ? 'btn-outline-primary' : 'btn-outline-secondary'} mark-read">
                                    ${notification.status === 'unread' ? 'Mark Read' : 'Read'}
                                </button>
                                <button class="btn btn-sm btn-outline-info archive-notification">
                                    <i class="ti ti-archive"></i> Archive
                                </button>
                            ` : `
                                <button class="btn btn-sm btn-outline-success restore-notification">
                                    <i class="ti ti-refresh"></i> Restore
                                </button>
                            `}
                            
                            <button class="btn btn-sm btn-outline-danger delete-notification">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    function updateUnreadCount() {
        unreadCount = notifications.filter(n => n.status === 'unread').length;
        unreadBadge.text(unreadCount).toggle(unreadCount > 0);
    }

    function setupSocketListeners() {
        // Initial notifications
        socket.on('initialNotifications', (initialNotifications) => {
            notifications = initialNotifications;
            renderNotifications();
            updateUnreadCount();
        });

        // New notification
        socket.on('newNotification', (newNotification) => {
            console.log('Received new notification:', newNotification);
            notifications.unshift(newNotification);
            notificationList.prepend(createNotificationElement(newNotification));
            updateUnreadCount();
            showToastNotification(newNotification);
        });

        // Notification updated
        socket.on('notificationUpdated', (updatedNotification) => {
            const index = notifications.findIndex(n => n._id === updatedNotification._id);
            if (index !== -1) {
                notifications[index] = updatedNotification;
                $(`.notification-item[data-id="${updatedNotification._id}"]`)
                    .removeClass('unread')
                    .addClass('read')
                    .find('.mark-read')
                    .removeClass('btn-outline-primary')
                    .addClass('btn-outline-secondary')
                    .text('Read');
                updateUnreadCount();
            }
        });

        // Handle archive events
        socket.on('notificationArchived', (notificationId) => {
            $(`.notification-item[data-id="${notificationId}"]`).remove();
            updateUnreadCount();
        });

        socket.on('allNotificationsArchived', () => {
            notificationList.empty();
            unreadCount = 0;
            unreadBadge.text(unreadCount).toggle(unreadCount > 0);
            showSuccessToast('All notifications archived');
        });

        socket.on('notificationRestored', (notification) => {
            $(`.notification-item[data-id="${notification._id}"]`).remove();
            showSuccessToast('Notification restored');
        });

        socket.on('notificationDeleted', (notificationId) => {
            notifications = notifications.filter(n => n._id !== notificationId);
            $(`.notification-item[data-id="${notificationId}"]`).remove();
            updateUnreadCount();
        });
    }

    function setupEventHandlers() {
        // Mark as read
        notificationList.on('click', '.mark-read', function() {
            const notificationId = $(this).closest('.notification-item').data('id');
            socket.emit('markAsRead', notificationId);
        });

        // Mark all as read
        markAllReadBtn.on('click', function() {
            $.ajax({
                url: '/api/notifications/read-all',
                method: 'PATCH',
                headers: { 'Authorization': 'Bearer ' + token },
                success: function() {
                    notifications.forEach(n => n.status = 'read');
                    $('.notification-item').removeClass('unread').addClass('read');
                    $('.mark-read')
                        .removeClass('btn-outline-primary')
                        .addClass('btn-outline-secondary')
                        .text('Read');
                    updateUnreadCount();
                    showSuccessToast('All notifications marked as read');
                },
                error: function(xhr) {
                    console.error('Error marking all as read:', xhr.responseText);
                    showErrorToast('Failed to mark all as read');
                }
            });
        });

        // Archive notification
        notificationList.on('click', '.archive-notification', function() {
            const notificationId = $(this).closest('.notification-item').data('id');
            archiveNotification(notificationId);
        });

        // Restore notification
        notificationList.on('click', '.restore-notification', function() {
            const notificationId = $(this).closest('.notification-item').data('id');
            restoreNotification(notificationId);
        });

        // Archive all
        archiveAllBtn.on('click', function() {
            if (confirm('Are you sure you want to archive all notifications?')) {
                archiveAllNotifications();
            }
        });

        // Delete notification
        notificationList.on('click', '.delete-notification', function(e) {
            e.stopPropagation();
            const notificationId = $(this).closest('.notification-item').data('id');
            deleteNotification(notificationId);
        });

        // Load more
        loadMoreBtn.on('click', function() {
            loadNotifications(currentTab === 'archived' ? 'archived' : null);
        });

        // Tab switching
        allTab.on('click', function() {
            currentTab = 'all';
            offset = 0;
            loadNotifications();
        });

        unreadTab.on('click', function() {
            currentTab = 'unread';
            offset = 0;
            loadNotifications('unread');
        });

        archivedTab.on('click', function() {
            currentTab = 'archived';
            offset = 0;
            loadNotifications('archived');
        });
    }

    function archiveNotification(notificationId) {
        $.ajax({
            url: `/api/notifications/${notificationId}/archive`,
            method: 'PATCH',
            headers: { 'Authorization': 'Bearer ' + token },
            success: function(response) {
                $(`.notification-item[data-id="${notificationId}"]`).remove();
                showSuccessToast('Notification archived');
                updateUnreadCount();
            },
            error: function(xhr) {
                console.error('Archive error:', xhr.responseText);
                showErrorToast('Failed to archive notification');
            }
        });
    }

    function archiveAllNotifications() {
        $.ajax({
            url: '/api/notifications/archive-all',
            method: 'PATCH',
            headers: { 'Authorization': 'Bearer ' + token },
            success: function(response) {
                notificationList.empty();
                unreadCount = 0;
                unreadBadge.text(unreadCount).toggle(unreadCount > 0);
                showSuccessToast(`Archived ${response.archivedCount} notifications`);
            },
            error: function(xhr) {
                console.error('Archive all error:', xhr.responseText);
                showErrorToast('Failed to archive all notifications');
            }
        });
    }

    function restoreNotification(notificationId) {
        $.ajax({
            url: `/api/notifications/${notificationId}/restore`,
            method: 'PATCH',
            headers: { 'Authorization': 'Bearer ' + token },
            success: function(response) {
                $(`.notification-item[data-id="${notificationId}"]`).remove();
                showSuccessToast('Notification restored');
            },
            error: function(xhr) {
                console.error('Restore error:', xhr.responseText);
                showErrorToast('Failed to restore notification');
            }
        });
    }

    function deleteNotification(notificationId) {
        if (!confirm('Are you sure you want to delete this notification?')) return;

        $.ajax({
            url: `/api/notifications/${notificationId}`,
            method: 'DELETE',
            headers: { 'Authorization': 'Bearer ' + token },
            success: function() {
                notifications = notifications.filter(n => n._id !== notificationId);
                $(`.notification-item[data-id="${notificationId}"]`).remove();
                updateUnreadCount();
                showSuccessToast('Notification deleted');
            },
            error: function(xhr) {
                console.error('Delete error:', xhr.responseText);
                showErrorToast('Failed to delete notification');
            }
        });
    }

    function showToastNotification(notification) {
        Toastify({
            text: `${notification.title}: ${notification.message}`,
            duration: 5000,
            gravity: "bottom",
            position: "right",
            backgroundColor: notification.type === 'alert' ? '#dc3545' : '#28a745',
            onClick: function() {
                if (notification.link) {
                    window.location.href = notification.link;
                }
            }
        }).showToast();
    }
});