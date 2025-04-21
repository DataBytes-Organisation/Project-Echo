$(document).ready(function() {
    // Dummy notifications for testing
    let notifications = [
        {
            id: 1,
            message: "New donation received from Jon Doe",
            date: new Date(),
            read: false,
            icon: "ti ti-receipt-2",
            type: "donation"
        },
        {
            id: 2,
            message: "New request from Jon Doe",
            date: new Date(Date.now() - 3600000), // 1 hour ago
            read: false,
            icon: "ti ti-alert-circle",
            type: "request"
        },
        {
            id: 3,
            message: "New user registration: jondoe@example.com",
            date: new Date(Date.now() - 172800000), // 2 days ago
            read: false,
            icon: "ti ti-user",
            type: "user"
        }
    ];

    // Render notifications
    function renderNotifications() {
        const notificationContainer = $("#notification-list");
        notificationContainer.empty();

        if (notifications.length === 0) {
            notificationContainer.html(`
                <div class="empty-notifications">
                    <i class="far fa-bell-slash"></i>
                    <h5>No notifications</h5>
                    <p>You're all caught up!</p>
                </div>
            `);
            return;
        }

        // Sort by date (newest first)
        notifications.sort((a, b) => b.date - a.date);

        notifications.forEach(notification => {
            const timeAgo = moment(notification.date).fromNow();
            const formattedDate = moment(notification.date).format('MMM D, YYYY h:mm A');
            const notificationItem = `
                <div class="d-flex align-items-start notification-item ${notification.read ? 'read' : 'unread'}" data-id="${notification.id}">
                    <div class="notification-icon">
                        <i class="${notification.icon}"></i>
                    </div>
                    <div class="notification-content">
                        <p class="notification-message mb-1">${notification.message}</p>
                        <p class="notification-date mb-2"><small><i class="far fa-clock me-1"></i>${timeAgo} (${formattedDate})</small></p>
                        <div class="notification-actions">
                            <button class="btn btn-sm ${notification.read ? 'btn-outline-secondary' : 'btn-outline-primary'} mark-read" ${notification.read ? 'disabled' : ''}>
                                <i class="fas fa-check me-1"></i>${notification.read ? 'Read' : 'Mark Read'}
                            </button>
                            <button class="btn btn-sm btn-outline-danger delete-notification">
                                <i class="fas fa-trash-alt me-1"></i>Delete
                            </button>
                        </div>
                    </div>
                </div>
            `;
            notificationContainer.append(notificationItem);
        });
    }

    // Mark notification as read
    $(document).on("click", ".mark-read", function () {
        const notificationId = $(this).closest(".notification-item").data("id");
        const notification = notifications.find(n => n.id === notificationId);
        if (notification) {
            notification.read = true;
            renderNotifications();
        }
    });

    // Delete notification
    $(document).on("click", ".delete-notification", function () {
        const notificationId = $(this).closest(".notification-item").data("id");
        notifications = notifications.filter(n => n.id !== notificationId);
        renderNotifications();
    });

    // Mark all as read
    $("#mark-all-read").on("click", function () {
        notifications.forEach(notification => {
            notification.read = true;
        });
        renderNotifications();
    });

    // Delete all read notifications
    $("#delete-all-read").on("click", function () {
        notifications = notifications.filter(n => !n.read);
        renderNotifications();
    });

    renderNotifications();
});
