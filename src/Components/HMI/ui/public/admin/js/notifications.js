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
    });

    socket.on('disconnect', (reason) => {
        console.log('Socket.IO disconnected:', reason);
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
    let archivedNotifications = [];
    let unreadCount = 0;
    let currentTab = 'all'; // 'all', 'unread', 'archived'

    // DOM Elements
    const notificationList = $("#notification-list");
    const unreadNotificationList = $("#unread-notification-list");
    const archivedNotificationList = $("#archived-notification-list");
    const unreadBadge = $("#unread-badge");
    const markAllReadBtn = $("#mark-all-read");
    const loadMoreBtn = $("#load-more");
    const archiveAllBtn = $("#archive-all");

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
                if (status === 'archived') {
                    if (offset === 0) {
                        archivedNotifications = response.notifications || [];
                    } else {
                        archivedNotifications = archivedNotifications.concat(response.notifications || []);
                    }
                    renderArchivedNotifications();
                } else {
                    if (offset === 0) {
                        notifications = response.notifications || [];
                    } else {
                        notifications = notifications.concat(response.notifications || []);
                    }
                    renderNotifications();
                    updateUnreadCount();
                }

                offset += (response.notifications || []).length;
            },
            error: function(xhr) {
                console.error('API Error:', xhr.responseText);
                showErrorToast('Failed to load notifications');
            },
            complete: function() {
                isLoading = false;
                loadMoreBtn.prop('disabled', false).html('Load More');

                const currentNotifications = status === 'archived' ? archivedNotifications : notifications;
                if (currentNotifications.length > 0 && currentNotifications.length % limit !== 0) {
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
        unreadNotificationList.empty();

        if (notifications.length === 0) {
            notificationList.html(`
                <div class="empty-notifications">
                    <i class="far fa-bell-slash"></i>
                    <h5>No notifications yet</h5>
                    <p>You're all caught up!</p>
                </div>
            `);
            unreadNotificationList.html(`
                <div class="empty-notifications">
                    <i class="far fa-bell-slash"></i>
                    <h5>No unread notifications</h5>
                    <p>You're all caught up!</p>
                </div>
            `);
            return;
        }

        // Filter unread notifications
        const unreadNotifications = notifications.filter(n => n.status === 'unread');

        notifications.forEach(notification => {
            notificationList.append(createNotificationElement(notification, false));
        });

        unreadNotifications.forEach(notification => {
            unreadNotificationList.append(createNotificationElement(notification, false));
        });
    }

    function renderArchivedNotifications() {
        console.log('Rendering archived notifications:', archivedNotifications);
        archivedNotificationList.empty();

        if (archivedNotifications.length === 0) {
            archivedNotificationList.html(`
                <div class="empty-notifications">
                    <i class="far fa-archive"></i>
                    <h5>No archived notifications</h5>
                    <p>Your archive is empty!</p>
                </div>
            `);
            return;
        }

        archivedNotifications.forEach(notification => {
            archivedNotificationList.append(createNotificationElement(notification, true));
        });
    }

    function createNotificationElement(notification, isArchivedView = false) {
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
        // Initial notifications (non-archived only)
        socket.on('initialNotifications', (initialNotifications) => {
            // Filter out archived notifications from initial load
            notifications = initialNotifications.filter(n => n.status !== 'archived');
            renderNotifications();
            updateUnreadCount();
        });

        // New notification (should never be archived initially)
        socket.on('newNotification', (newNotification) => {
            console.log('Received new notification:', newNotification);
            notifications.unshift(newNotification);

            if (currentTab === 'all') {
                notificationList.prepend(createNotificationElement(newNotification, false));
            } else if (currentTab === 'unread' && newNotification.status === 'unread') {
                unreadNotificationList.prepend(createNotificationElement(newNotification, false));
            }

            updateUnreadCount();
            showToastNotification(newNotification);
        });

        // Notification updated
        socket.on('notificationUpdated', (updatedNotification) => {
            const index = notifications.findIndex(n => n._id === updatedNotification._id);
            if (index !== -1) {
                notifications[index] = updatedNotification;

                // Update UI based on current tab
                if (currentTab === 'all') {
                    $(`.notification-item[data-id="${updatedNotification._id}"]`)
                        .replaceWith(createNotificationElement(updatedNotification, false));
                } else if (currentTab === 'unread') {
                    if (updatedNotification.status === 'unread') {
                        $(`.notification-item[data-id="${updatedNotification._id}"]`)
                            .replaceWith(createNotificationElement(updatedNotification, false));
                    } else {
                        $(`.notification-item[data-id="${updatedNotification._id}"]`).remove();
                    }
                }

                updateUnreadCount();
            }
        });

        // Handle archive events
        socket.on('notificationArchived', (notificationId) => {
            // Remove from active notifications and add to archived
            const archivedIndex = notifications.findIndex(n => n._id === notificationId);
            if (archivedIndex !== -1) {
                const archivedNotification = notifications[archivedIndex];
                archivedNotification.status = 'archived';
                archivedNotifications.unshift(archivedNotification);
                notifications.splice(archivedIndex, 1);

                // Update UI
                $(`.notification-item[data-id="${notificationId}"]`).remove();

                if (currentTab === 'archived') {
                    archivedNotificationList.prepend(createNotificationElement(archivedNotification, true));
                }

                updateUnreadCount();
            }
        });

        socket.on('allNotificationsArchived', () => {
            // Move all non-archived notifications to archived
            const toArchive = notifications.filter(n => n.status !== 'archived');
            archivedNotifications = toArchive.concat(archivedNotifications);
            notifications = notifications.filter(n => n.status === 'archived');

            // Update UI
            notificationList.empty();
            unreadNotificationList.empty();
            unreadCount = 0;
            unreadBadge.text(unreadCount).toggle(unreadCount > 0);

            if (currentTab === 'archived') {
                renderArchivedNotifications();
            }

            showSuccessToast('All notifications archived');
        });

        socket.on('notificationRestored', (notification) => {
            // Remove from archived and add to active notifications
            const archivedIndex = archivedNotifications.findIndex(n => n._id === notification._id);
            if (archivedIndex !== -1) {
                archivedNotifications.splice(archivedIndex, 1);
                notifications.unshift(notification);

                // Update UI
                $(`.notification-item[data-id="${notification._id}"]`).remove();

                if (currentTab === 'all') {
                    notificationList.prepend(createNotificationElement(notification, false));
                } else if (currentTab === 'unread' && notification.status === 'unread') {
                    unreadNotificationList.prepend(createNotificationElement(notification, false));
                }

                updateUnreadCount();
            }

            showSuccessToast('Notification restored');
        });

        socket.on('notificationDeleted', (notificationId) => {
            // Remove from both arrays
            notifications = notifications.filter(n => n._id !== notificationId);
            archivedNotifications = archivedNotifications.filter(n => n._id !== notificationId);

            // Update UI
            $(`.notification-item[data-id="${notificationId}"]`).remove();
            updateUnreadCount();
        });
    }

    function setupEventHandlers() {
        // Mark as read
        $(document).on('click', '.mark-read', function() {
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
                    notifications.forEach(n => {
                        if (n.status === 'unread') {
                            n.status = 'read';
                        }
                    });

                    // Update UI
                    $('.notification-item').each(function() {
                        const id = $(this).data('id');
                        const notification = notifications.find(n => n._id === id);
                        if (notification && notification.status === 'read') {
                            $(this)
                                .removeClass('unread')
                                .addClass('read')
                                .find('.mark-read')
                                .removeClass('btn-outline-primary')
                                .addClass('btn-outline-secondary')
                                .text('Read');
                        }
                    });

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
        $(document).on('click', '.archive-notification', function() {
            const notificationId = $(this).closest('.notification-item').data('id');
            archiveNotification(notificationId);
        });

        // Restore notification
        $(document).on('click', '.restore-notification', function() {
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
        $(document).on('click', '.delete-notification', function(e) {
            e.stopPropagation();
            const notificationId = $(this).closest('.notification-item').data('id');
            deleteNotification(notificationId);
        });

        // Load more
        loadMoreBtn.on('click', function() {
            if (currentTab === 'archived') {
                loadNotifications('archived');
            } else if (currentTab === 'unread') {
                loadNotifications('unread');
            } else {
                loadNotifications();
            }
        });

        // Tab switching
        allTab.on('click', function() {
            currentTab = 'all';
            offset = 0;
            // Show the correct tab content using Bootstrap
            $('#all').addClass('show active');
            $('#unread').removeClass('show active');
            $('#archived').removeClass('show active');

            // Load the correct data
            loadNotifications();
        });

        unreadTab.on('click', function() {
            currentTab = 'unread';
            offset = 0;
            // Show the correct tab content using Bootstrap
            $('#unread').addClass('show active');
            $('#all').removeClass('show active');
            $('#archived').removeClass('show active');

            // Load unread notifications
            loadNotifications('unread');
        });

        archivedTab.on('click', function() {
            currentTab = 'archived';
            offset = 0;
            // Show the correct tab content using Bootstrap
            $('#archived').addClass('show active');
            $('#all').removeClass('show active');
            $('#unread').removeClass('show active');

            // Load archived notifications
            loadNotifications('archived');
            console.log('Switched to archived tab, loading data...');
        });
    }

    function archiveNotification(notificationId) {
        $.ajax({
            url: `/api/notifications/${notificationId}/archive`,
            method: 'PATCH',
            headers: { 'Authorization': 'Bearer ' + token },
            success: function(response) {
                // Socket will handle the UI update via notificationArchived event
                showSuccessToast('Notification archived');
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
                // Socket will handle the UI update via allNotificationsArchived event
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
                // Socket will handle the UI update via notificationRestored event
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
                // Socket will handle the UI update via notificationDeleted event
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

    // Initialize tab visibility
    $('.tab-pane').removeClass('show active');
    $('#all').addClass('show active');
    notificationList.parent().show();
});