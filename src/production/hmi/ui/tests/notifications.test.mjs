"use strict";

/**
 * FR-D2 regression tests for the admin notification feed.
 *
 * These import the real services/notifications.js rather than restating its
 * logic. The shaping and the read/deleted join were deliberately kept out of
 * server.js so they could be exercised here without Mongo or a running server.
 *
 * The browser side (safe DOM rendering, badge) is covered separately: see the
 * PR description for the XSS payload check against the live stack, which is a
 * stronger check than a shimmed DOM would give.
 */

import test from "node:test";
import assert from "node:assert/strict";

import notificationFeed from "../services/notifications.js";

const { buildFeed, applyState } = notificationFeed;

const donation = (overrides = {}) => ({
  _id: "d1",
  created: 1743705600,
  billing_details: { email: "john@icloud.com" },
  ...overrides
});

const user = (overrides = {}) => ({
  _id: "u1",
  email: "someone@example.com",
  createdAt: "2026-08-01T00:00:00.000Z",
  ...overrides
});

test("buildFeed derives ids from the source record so state survives a reload", () => {
  const feed = buildFeed({ donations: [donation()], users: [user()] });

  assert.deepEqual(
    feed.map(item => item.id),
    ["donation:d1", "user:u1"]
  );
});

test("buildFeed reads the donor email out of billing_details", () => {
  const [item] = buildFeed({ donations: [donation()] });

  assert.equal(item.message, "New donation received from john@icloud.com");
  assert.equal(item.type, "donation");
});

test("buildFeed falls back when a donation has no email attached", () => {
  const [item] = buildFeed({ donations: [donation({ billing_details: {} })] });

  assert.equal(item.message, "New donation received from an anonymous supporter");
});

test("buildFeed converts the Stripe seconds timestamp to milliseconds", () => {
  const [item] = buildFeed({ donations: [donation({ created: 1743705600 })] });

  assert.equal(item.date.getTime(), 1743705600 * 1000);
});

test("buildFeed falls back to the username when a user has no email", () => {
  const [item] = buildFeed({ users: [user({ email: undefined, username: "jdoe" })] });

  assert.equal(item.message, "New user registration: jdoe");
});

test("buildFeed copes with neither email nor username", () => {
  const [item] = buildFeed({ users: [user({ email: undefined, username: undefined })] });

  assert.equal(item.message, "New user registration: unknown account");
});

test("buildFeed returns an empty feed rather than throwing when given nothing", () => {
  assert.deepEqual(buildFeed(), []);
  assert.deepEqual(buildFeed({}), []);
});

test("applyState marks an entry read when a stored row says so", () => {
  const feed = buildFeed({ donations: [donation()] });
  const { notifications } = applyState(feed, [{ _id: "donation:d1", read: true }]);

  assert.equal(notifications[0].read, true);
});

test("applyState treats anything with no stored row as unread", () => {
  const feed = buildFeed({ donations: [donation()] });
  const { notifications } = applyState(feed, []);

  assert.equal(notifications[0].read, false);
});

test("applyState drops deleted entries entirely rather than flagging them", () => {
  const feed = buildFeed({ donations: [donation({ _id: "d1" }), donation({ _id: "d2" })] });
  const { notifications } = applyState(feed, [{ _id: "donation:d1", deleted: true }]);

  assert.deepEqual(notifications.map(item => item.id), ["donation:d2"]);
});

test("a deleted entry is not counted as unread", () => {
  const feed = buildFeed({ donations: [donation({ _id: "d1" }), donation({ _id: "d2" })] });
  const { unreadCount } = applyState(feed, [{ _id: "donation:d1", deleted: true }]);

  assert.equal(unreadCount, 1);
});

test("unreadCount is derived from the same list it is counting", () => {
  const feed = buildFeed({
    donations: [donation({ _id: "d1" }), donation({ _id: "d2" }), donation({ _id: "d3" })]
  });

  const { notifications, unreadCount } = applyState(feed, [
    { _id: "donation:d1", read: true },
    { _id: "donation:d2", deleted: true }
  ]);

  assert.equal(unreadCount, notifications.filter(item => !item.read).length);
  assert.equal(unreadCount, 1);
});

test("marking everything read leaves a zero unread count", () => {
  const feed = buildFeed({ donations: [donation({ _id: "d1" })], users: [user({ _id: "u1" })] });

  const { unreadCount } = applyState(feed, [
    { _id: "donation:d1", read: true },
    { _id: "user:u1", read: true }
  ]);

  assert.equal(unreadCount, 0);
});

test("applyState returns newest first", () => {
  const feed = buildFeed({
    donations: [
      donation({ _id: "old", created: 1000 }),
      donation({ _id: "new", created: 2000 })
    ]
  });

  const { notifications } = applyState(feed, []);

  assert.deepEqual(notifications.map(item => item.id), ["donation:new", "donation:old"]);
});

test("applyState ignores stored rows for records that no longer exist", () => {
  const feed = buildFeed({ donations: [donation({ _id: "d1" })] });
  const { notifications, unreadCount } = applyState(feed, [
    { _id: "donation:long-gone", read: true, deleted: true }
  ]);

  assert.equal(notifications.length, 1);
  assert.equal(unreadCount, 1);
});
