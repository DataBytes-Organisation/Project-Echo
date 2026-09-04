// =============================================================
// Admin notifications
//
// Loads the feed from /api/notifications and sends every action straight back
// to the server, so read, delete and mark-all-read all survive a refresh.
//
// Nothing here builds an element out of an HTML string. Notification text comes
// from donation and account records, which means it is effectively user
// supplied, and dropping that into innerHTML would let a crafted email address
// run script in the admin console. Everything is created with the DOM API and
// written in with textContent instead.
// =============================================================

import {
  retrieveNotifications,
  markNotificationRead,
  markAllNotificationsRead,
  deleteNotification,
  deleteReadNotifications
} from "/js/routes.js";
import { getApiErrorMessage } from "/js/HMI-utils.js";

const pageState = window.createAdminPageState ? window.createAdminPageState() : null;

// The single copy of what the server last told us. The list and the badge are
// both drawn from this, so they cannot disagree.
let notifications = [];
let unreadCount = 0;

// Icons are chosen here from the notification type rather than taken from the
// response, so nothing outside this file can decide what class lands on an
// element.
const ICONS_BY_TYPE = {
  donation: "ti ti-receipt-2",
  request: "ti ti-alert-circle",
  user: "ti ti-user"
};

const DEFAULT_ICON = "ti ti-bell";

function createElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function formatWhen(rawDate) {
  const date = new Date(rawDate);
  if (Number.isNaN(date.getTime())) return "";

  if (typeof window.moment === "function") {
    const when = window.moment(date);
    return `${when.fromNow()} (${when.format("MMM D, YYYY h:mm A")})`;
  }

  return date.toLocaleString();
}

function buildEmptyState() {
  const wrapper = createElement("div", "empty-notifications");
  wrapper.appendChild(createElement("i", "far fa-bell-slash"));
  wrapper.appendChild(createElement("h5", null, "No notifications"));
  wrapper.appendChild(createElement("p", null, "You're all caught up!"));
  return wrapper;
}

function buildNotificationItem(notification) {
  const item = createElement(
    "div",
    `d-flex align-items-start notification-item ${notification.read ? "read" : "unread"}`
  );
  item.dataset.id = notification.id;

  const iconWrapper = createElement("div", "notification-icon");
  iconWrapper.appendChild(
    createElement("i", ICONS_BY_TYPE[notification.type] || DEFAULT_ICON)
  );
  item.appendChild(iconWrapper);

  const content = createElement("div", "notification-content");

  // textContent, so a message containing markup is shown as text rather than parsed
  content.appendChild(
    createElement("p", "notification-message mb-1", notification.message)
  );

  const dateLine = createElement("p", "notification-date mb-2");
  const dateSmall = createElement("small");
  dateSmall.appendChild(createElement("i", "far fa-clock me-1"));
  dateSmall.appendChild(document.createTextNode(formatWhen(notification.date)));
  dateLine.appendChild(dateSmall);
  content.appendChild(dateLine);

  const actions = createElement("div", "notification-actions");

  const readButton = createElement(
    "button",
    `btn btn-sm ${notification.read ? "btn-outline-secondary" : "btn-outline-primary"} mark-read`
  );
  readButton.type = "button";
  readButton.disabled = Boolean(notification.read);
  readButton.appendChild(createElement("i", "fas fa-check me-1"));
  readButton.appendChild(
    document.createTextNode(notification.read ? "Read" : "Mark Read")
  );
  actions.appendChild(readButton);

  const deleteButton = createElement("button", "btn btn-sm btn-outline-danger delete-notification");
  deleteButton.type = "button";
  deleteButton.appendChild(createElement("i", "fas fa-trash-alt me-1"));
  deleteButton.appendChild(document.createTextNode("Delete"));
  actions.appendChild(deleteButton);

  content.appendChild(actions);
  item.appendChild(content);

  return item;
}

function renderList() {
  const container = document.getElementById("notification-list");
  if (!container) return;

  container.replaceChildren();

  if (notifications.length === 0) {
    container.appendChild(buildEmptyState());
    return;
  }

  for (const notification of notifications) {
    container.appendChild(buildNotificationItem(notification));
  }
}

// The header is injected after this script runs, so the badge may not exist yet.
// renderBadge is safe to call at any point and is re-run once the header lands.
function renderBadge() {
  const badge = document.getElementById("notification-badge");
  if (!badge) return;

  badge.textContent = unreadCount > 0 ? String(unreadCount) : "";
  badge.hidden = unreadCount === 0;
  badge.setAttribute(
    "aria-label",
    unreadCount === 1 ? "1 unread notification" : `${unreadCount} unread notifications`
  );
}

// Only place that draws anything, so the badge can never fall out of step with
// the list it is counting.
function render() {
  renderList();
  renderBadge();
}

function applyResult(data) {
  notifications = Array.isArray(data && data.notifications) ? data.notifications : [];
  unreadCount = Number.isFinite(data && data.unreadCount)
    ? data.unreadCount
    : notifications.filter(item => !item.read).length;
  render();
}

async function runAction(action, failureMessage) {
  if (pageState) {
    pageState.hideError();
    pageState.showLoading();
  }

  try {
    const response = await action();
    applyResult(response.data);
  } catch (error) {
    console.error(failureMessage, error);
    if (pageState) pageState.showError(getApiErrorMessage(error, failureMessage));
  } finally {
    if (pageState) pageState.hideLoading();
  }
}

function idFromEvent(event, selector) {
  const button = event.target.closest(selector);
  if (!button) return null;

  const item = button.closest(".notification-item");
  return item ? item.dataset.id : null;
}

function wireUp() {
  const container = document.getElementById("notification-list");
  if (container) {
    // delegated, so buttons rebuilt on every render stay wired up
    container.addEventListener("click", event => {
      const readId = idFromEvent(event, ".mark-read");
      if (readId) {
        runAction(() => markNotificationRead(readId), "Could not mark the notification as read.");
        return;
      }

      const deleteId = idFromEvent(event, ".delete-notification");
      if (deleteId) {
        runAction(() => deleteNotification(deleteId), "Could not delete the notification.");
      }
    });
  }

  const markAllButton = document.getElementById("mark-all-read");
  if (markAllButton) {
    markAllButton.addEventListener("click", () =>
      runAction(markAllNotificationsRead, "Could not mark all notifications as read.")
    );
  }

  const deleteReadButton = document.getElementById("delete-all-read");
  if (deleteReadButton) {
    deleteReadButton.addEventListener("click", () =>
      runAction(deleteReadNotifications, "Could not delete the read notifications.")
    );
  }

  // The header component is loaded in separately, so watch for it arriving and
  // paint the badge then rather than guessing at a delay.
  //
  // The badge itself sits inside #header, so the observer has to stop as soon as
  // it has what it wants. Leaving it connected means renderBadge's own write
  // counts as a mutation, which calls renderBadge again, and the page locks up.
  const header = document.getElementById("header");
  if (header && !document.getElementById("notification-badge")) {
    const observer = new MutationObserver(() => {
      if (!document.getElementById("notification-badge")) return;
      observer.disconnect();
      renderBadge();
    });
    observer.observe(header, { childList: true, subtree: true });
  }
}

wireUp();
runAction(retrieveNotifications, "Could not load notifications.");

export { buildNotificationItem, renderBadge, applyResult };
