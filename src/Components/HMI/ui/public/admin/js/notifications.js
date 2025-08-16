$(document).ready(function() {
    // Verify Socket.IO is available
    if (typeof io === 'undefined') {
        console.error('Socket.IO not loaded!');
        return;
    }

    const token = localStorage.getItem('token');
    console.log('Current token:', token);

    if (!token) {
        console.error('No token found in localStorage');
        // Redirect to login or handle missing token
        window.location.href = '/login';
        return;
    }
    // Connect to Socket.io
    const socket = io('http://localhost:8080', {
        auth: {
            token: localStorage.getItem('token')
        },
        transports: ['websocket', 'polling'], // Add fallback transport
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
        withCredentials: true
    });

    // Connection status logging
    socket.on('connect', () => {
        console.log('Socket.IO connected with ID:', socket.id);
    });

    socket.on('disconnect', (reason) => {
        console.log('Socket.IO disconnected:', reason);
        if (reason === 'io server disconnect') {
            // Try to reconnect after getting new credentials if needed
            socket.connect();
        }
    });

    socket.on('connect_error', (err) => {
        console.error('Socket.IO connection error:', err.message);
        // Try to reconnect with fresh token if auth fails
        if (err.message.includes('auth')) {
            socket.auth.token = localStorage.getItem('token');
            socket.connect();
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
            headers: {
                'Authorization': 'Bearer ' + localStorage.getItem('token'),
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

                // Only hide if we know we've got all notifications
                if (notifications.length > 0 && notifications.length % limit !== 0) {
                    loadMoreBtn.hide();
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
        // Connection status indicator
        const updateConnectionStatus = (connected) => {
            const indicator = $('#connection-indicator');
            indicator.removeClass(connected ? 'disconnected' : 'connected')
                .addClass(connected ? 'connected' : 'disconnected')
                .text(connected ? 'Connected' : 'Disconnected');
        };

        socket.on('connect', () => updateConnectionStatus(true));
        socket.on('disconnect', () => updateConnectionStatus(false));

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