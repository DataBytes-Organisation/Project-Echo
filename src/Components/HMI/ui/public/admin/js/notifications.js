$(document).ready(function() {
    // Connect to Socket.io
    const socket = io({
        auth: {
            token: localStorage.getItem('token') // Assuming JWT is stored here
        }
    });

    let notifications = [];
    let unreadCount = 0;

    // DOM Elements
    const notificationList = $("#notification-list");
    const unreadBadge = $("#unread-badge");
    const markAllReadBtn = $("#mark-all-read");
    const loadMoreBtn = $("#load-more");
    let isLoading = false;
    let offset = 0;
    const limit = 10;

    // Initialize
    loadNotifications();
    setupSocketListeners();
    setupEventHandlers();

    function loadNotifications() {
        if (isLoading) return;

        isLoading = true;
        loadMoreBtn.prop('disabled', true).html('<i class="fas fa-spinner fa-spin"></i> Loading...');

        $.ajax({
            url: `/api/notifications?limit=${limit}&offset=${offset}`,
            headers: { 'Authorization': 'Bearer ' + localStorage.getItem('token') },
            success: function(response) {
                if (offset === 0) {
                    notifications = response.notifications;
                    renderNotifications();
                } else {
                    notifications = [...notifications, ...response.notifications];
                    appendNotifications(response.notifications);
                }

                updateUnreadCount();
                offset += limit;

                if (offset >= response.total) {
                    loadMoreBtn.hide();
                }
            },
            complete: function() {
                isLoading = false;
                loadMoreBtn.prop('disabled', false).html('Load More');
            }
        });
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
        const isUnread = notification.status === 'unread';

        return `
            <div class="notification-item ${isUnread ? 'unread' : 'read'}" 
                 data-id="${notification._id}">
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
                            <button class="btn btn-sm ${isUnread ? 'btn-outline-primary' : 'btn-outline-secondary'} mark-read">
                                ${isUnread ? 'Mark Read' : 'Read'}
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
            notifications.unshift(newNotification);

            // If first page, prepend the new notification
            if (offset === 0) {
                notificationList.prepend(createNotificationElement(newNotification));
            }

            updateUnreadCount();

            // Show toast notification
            showToastNotification(newNotification);
        });

        // Notification updated (e.g., marked as read)
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
                headers: { 'Authorization': 'Bearer ' + localStorage.getItem('token') },
                success: function() {
                    notifications.forEach(n => n.status = 'read');
                    $('.notification-item').removeClass('unread').addClass('read');
                    $('.mark-read')
                        .removeClass('btn-outline-primary')
                        .addClass('btn-outline-secondary')
                        .text('Read');
                    updateUnreadCount();
                }
            });
        });

        // Load more
        loadMoreBtn.on('click', loadNotifications);
    }

    function showToastNotification(notification) {
        // Using Toastify for toast notifications
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