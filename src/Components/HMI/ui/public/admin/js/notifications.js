let socket = null;
let isConnected = false;

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
    console.log("Initial notifications:", notifications);
    console.log("Unread count:", notifications.filter(n => !n.read).length);

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
        // Initializing a Socket.IO connection）
     function initSocket() {
         console.log('🔍 Start initializing the socket connection...');
    
         socket = io('http://localhost:8081');
    
         socket.on('connect', () => {
             console.log('✅ Socket connection successful. Connection ID', socket.id);
             isConnected = true;
             updateConnectionStatus('connected');
             const userId = getUserId();
             if (userId) {
                 socket.emit('join-user-room', userId);
                 console.log(`📌 Joined user room: ${userId}`);
        }
    });
    
    // Connection error event (retain automatic reconnection logic)
    socket.on('connect_error', (error) => {
        console.error('❌ Socket connection failed:', error.message);
        console.error('Possible reasons: Wrong server address/CORS issue/Server not running');
        isConnected = false;
        updateConnectionStatus('disconnected');
        
        // Automatic reconnection mechanism (original logic retained)
        setTimeout(() => {
            if (!isConnected) {
                console.log('Attempting to reconnect...');
                initSocket();
            }
        }, 5000);
    });
    
    // Disconnect event
    socket.on('disconnect', (reason) => {
        console.log('❌ Socket disconnected, reason:', reason);
        isConnected = false;
        updateConnectionStatus('disconnected');
        
        if (reason === 'io server disconnect') {
            socket.connect();
        }
    });
    
     socket.on('new-notification', function(notification) {
         console.log('New notification received:', notification);
        
        notifications.unshift({
            id: notification._id || Date.now(),
            message: notification.message,
            date: new Date(notification.createdAt || Date.now()),
            read: false,
            icon: getIconForType(notification.type),
            type: notification.type
        });
        
        saveNotificationsToLocalStorage();
        renderNotifications();
        updateNotificationBadge();
        showToastNotification(notification.message);
    });
    
    return socket;
}
    
    // Update the connection status UI
    function updateConnectionStatus(status) {
        const $status = $('#connection-status');
        const $indicator = $status.find('.status-indicator');
        const $text = $status.find('.status-text');

        $status.removeClass('connected disconnected connecting').addClass(status);
        $indicator.removeClass('status-connected status-disconnected status-connecting').addClass(`status-${status}`);
        $text.text(status === 'connected' ? 'Connected' : status === 'connecting' ? 'Connecting...' : 'Disconnected');
    }
    
    // Show toast notification
    function showToastNotification(message) {
        $('#toast-message').text(message);
        $('#toast-notification').toast('show');
        
        // Add badge animation
        $("#notification-badge").addClass("pulse");
        setTimeout(() => {
            $("#notification-badge").removeClass("pulse");
        }, 1000);
    }
    
    //  Get user ID
    function getUserId() {
    // This should retrieve the user ID from the login state or localStorage
    // Temporarily return a fixed value for testing
        return "64bf20397e048a9822077b74";
    }
    
    // Get icon based on type
    function getIconForType(type) {
        const icons = {
            'donation': 'ti ti-receipt-2',
            'request': 'ti ti-alert-circle',
            'user': 'ti ti-user',
            'system': 'ti ti-bell',
            'default': 'ti ti-bell'
        };
        return icons[type] || 'ti ti-bell';
    }
    
    // Test notification
    function testNotification() {
        fetch('http://localhost:8080/test-notification', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                userId: getUserId(),
                title: "Test Notification",
                message: "This is a test notification from the UI"
            })
        })
        .then(response => response.json())
        .then(data => console.log('Test notification created:', data))
        .catch(error => console.error('Error:', error));
    }
    //Function to update notification permissions (display unread count)
    function updateNotificationBadge() {
        const unreadCount = notifications.filter(notification => !notification.read).length;
        const badge = $("#notification-badge");    
        if (unreadCount > 0) {
            badge.text(unreadCount);
            badge.show();
        } else {
            badge.hide();
        }
    }
    
    // Mark notification as read
    $(document).on("click", ".mark-read", function () {
        const notificationId = $(this).closest(".notification-item").data("id");
        const notification = notifications.find(n => n.id === notificationId);
        if (notification) {
            notification.read = true;
            renderNotifications();
        }
        updateNotificationBadge();
    });

    // Delete notification
    $(document).on("click", ".delete-notification", function () {
        const notificationId = $(this).closest(".notification-item").data("id");
        notifications = notifications.filter(n => n.id !== notificationId);
        renderNotifications();
        updateNotificationBadge();
    });

    // Mark all as read
    $("#mark-all-read").on("click", function () {
        notifications.forEach(notification => {
            notification.read = true;
        });
        renderNotifications();
        updateNotificationBadge();
    });

    // Delete all read notifications
    $("#delete-all-read").on("click", function () {
        notifications = notifications.filter(n => !n.read);
        renderNotifications();
        updateNotificationBadge();
    });

    renderNotifications();
    updateNotificationBadge();
    setTimeout(function() {
    updateNotificationBadge();
    }, 100);
});
