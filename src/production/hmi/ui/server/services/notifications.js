"use strict";

/**
 * Admin notification feed assembly.
 *
 * The admin notification list used to be a hardcoded array in the browser, so
 * nothing survived a refresh. The feed is now derived from records we already
 * hold (donations and user registrations) and the read/deleted state is stored
 * separately, keyed by a stable id per source record.
 *
 * The pure parts live here rather than in server.js so they can be tested
 * without booting the server or touching Mongo. server.js does the reads and
 * hands the documents in.
 */

const ANONYMOUS_DONOR = "an anonymous supporter";
const UNKNOWN_ACCOUNT = "unknown account";

/**
 * Build notification entries out of the source records.
 *
 * The id is derived from the source document rather than generated, otherwise
 * the read state would not line up with the same notification on the next
 * request and everything would look unread again after a reload.
 *
 * @param {{donations?: object[], users?: object[]}} sources
 * @returns {object[]}
 */
function buildFeed({ donations = [], users = [] } = {}) {
  const feed = [];

  for (const donation of donations) {
    const email = donation.billing_details && donation.billing_details.email;
    feed.push({
      id: `donation:${donation._id}`,
      type: "donation",
      message: `New donation received from ${email || ANONYMOUS_DONOR}`,
      // Stripe records seconds, JavaScript wants milliseconds
      date: donation.created ? new Date(donation.created * 1000) : new Date(0)
    });
  }

  for (const user of users) {
    feed.push({
      id: `user:${user._id}`,
      type: "user",
      message: `New user registration: ${user.email || user.username || UNKNOWN_ACCOUNT}`,
      date: user.createdAt ? new Date(user.createdAt) : new Date(0)
    });
  }

  return feed;
}

/**
 * Join the stored read/deleted flags onto the feed.
 *
 * Deleted entries are dropped rather than returned with a flag, so the browser
 * never has to know they existed. unreadCount is derived from the same list it
 * is counting, which is what keeps the badge and the list from disagreeing.
 *
 * @param {object[]} feed
 * @param {object[]} stateRows  Documents of { _id, read, deleted }.
 * @returns {{notifications: object[], unreadCount: number}}
 */
function applyState(feed = [], stateRows = []) {
  const byId = new Map();
  for (const row of stateRows) {
    byId.set(row._id, row);
  }

  const notifications = feed
    .filter(item => !(byId.get(item.id) || {}).deleted)
    .map(item => ({ ...item, read: Boolean((byId.get(item.id) || {}).read) }))
    .sort((a, b) => b.date - a.date);

  return {
    notifications,
    unreadCount: notifications.filter(item => !item.read).length
  };
}

module.exports = { buildFeed, applyState };
