$(document).ready(function() {
    const token = localStorage.getItem('token');
    if (!token) {
        window.location.href = '/login';
        return;
    }

    // Load preferences
    loadPreferences(token);

    // Save button handler
    $('#saveButton').on('click', function() {
        savePreferences(token);
    });

    // Reset button handler
    $('#resetButton').on('click', function() {
        resetToDefaults(token);
    });

    // DND toggle handler
    $('#enableDND').on('change', function() {
        $('#dndSettings').toggle(this.checked);
    });
});

function loadPreferences(token) {
    $.ajax({
        url: '/api/notification-preferences',
        method: 'GET',
        headers: {
            'Authorization': 'Bearer ' + token,
            'Content-Type': 'application/json'
        },
        success: function(response) {
            console.log(response);
            populatePreferencesForm(response);
        },
        error: function(xhr) {
            console.error('Error loading preferences:', xhr);
            alert('Error loading preferences. Please try again.');
        }
    });
    console.log("Loading preferences");
}

function savePreferences(token) {
    const preferencesData = {
        channels: {
            inApp: $('#inAppNotifications').is(':checked'),
            email: $('#emailNotifications').is(':checked'),
            push: $('#pushNotifications').is(':checked')
        },
        types: {
            system: $('#systemNotifications').is(':checked'),
            donation: $('#donationNotifications').is(':checked'),
            request: $('#requestNotifications').is(':checked'),
            user: $('#userNotifications').is(':checked')
        },
        doNotDisturb: {
            enabled: $('#enableDND').is(':checked'),
            startTime: $('#dndStartTime').val(),
            endTime: $('#dndEndTime').val(),
            days: getSelectedDays()
        }
    };

    $.ajax({
        url: '/api/notification-preferences',
        method: 'PUT',
        headers: {
            'Authorization': 'Bearer ' + token,
            'Content-Type': 'application/json'
        },
        data: JSON.stringify(preferencesData),
        success: function(response) {
            alert('Preferences saved successfully!');
        },
        error: function(xhr) {
            console.error('Error saving preferences:', xhr);
            alert('Error saving preferences. Please try again.');
        }
    });
}

function resetToDefaults(token) {
    if (confirm('Are you sure you want to reset to default preferences?')) {
        const defaultPreferences = {
            channels: {
                inApp: true,
                email: false,
                push: false
            },
            types: {
                system: true,
                donation: true,
                request: true,
                user: false
            },
            doNotDisturb: {
                enabled: false,
                startTime: '22:00',
                endTime: '07:00',
                days: [0, 1, 2, 3, 4, 5, 6]
            }
        };

        $.ajax({
            url: '/api/notification-preferences',
            method: 'PUT',
            headers: {
                'Authorization': 'Bearer ' + token,
                'Content-Type': 'application/json'
            },
            data: JSON.stringify(defaultPreferences),
            success: function(response) {
                populatePreferencesForm(response);
                alert('Preferences reset to defaults!');
            },
            error: function(xhr) {
                console.error('Error resetting preferences:', xhr);
                alert('Error resetting preferences. Please try again.');
            }
        });
    }
}

function populatePreferencesForm(preferences) {
    // Channels
    $('#inAppNotifications').prop('checked', preferences.preferences.channels.inApp);
    $('#emailNotifications').prop('checked', preferences.preferences.channels.email);
    $('#pushNotifications').prop('checked', preferences.preferences.channels.push);

    // Types
    $('#systemNotifications').prop('checked', preferences.preferences.types.system);
    $('#donationNotifications').prop('checked', preferences.preferences.types.donation);
    $('#requestNotifications').prop('checked', preferences.preferences.types.request);
    $('#userNotifications').prop('checked', preferences.preferences.types.user);

    // DND
    $('#enableDND').prop('checked', preferences.preferences.doNotDisturb.enabled);
    $('#dndStartTime').val(preferences.preferences.doNotDisturb.startTime);
    $('#dndEndTime').val(preferences.preferences.doNotDisturb.endTime);

    // Set days
    setSelectedDays(preferences.preferences.doNotDisturb.days);
    $('#dndSettings').toggle(preferences.preferences.doNotDisturb.enabled);
}

function getSelectedDays() {
    const days = [];
    if ($('#dndMonday').is(':checked')) days.push(1);
    if ($('#dndTuesday').is(':checked')) days.push(2);
    if ($('#dndWednesday').is(':checked')) days.push(3);
    if ($('#dndThursday').is(':checked')) days.push(4);
    if ($('#dndFriday').is(':checked')) days.push(5);
    if ($('#dndSaturday').is(':checked')) days.push(6);
    if ($('#dndSunday').is(':checked')) days.push(0);
    return days;
}

function setSelectedDays(days) {
    $('#dndMonday').prop('checked', days.includes(1));
    $('#dndTuesday').prop('checked', days.includes(2));
    $('#dndWednesday').prop('checked', days.includes(3));
    $('#dndThursday').prop('checked', days.includes(4));
    $('#dndFriday').prop('checked', days.includes(5));
    $('#dndSaturday').prop('checked', days.includes(6));
    $('#dndSunday').prop('checked', days.includes(0));
}