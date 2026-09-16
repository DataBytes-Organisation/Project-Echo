"use strict";

/**
 * Tests for the JSON session guard used by the notification API.
 *
 * createApiSessionGuard takes the token lookup as an argument, so these run
 * against the real guard without Redis. The point of the guard is that it never
 * answers with a redirect: the page guard sends a browser to /login, and an API
 * caller following that redirect gets the login page back with a 200 and tries
 * to parse HTML as JSON.
 */

import test from "node:test";
import assert from "node:assert/strict";

import middleware from "../middleware/index.js";

const { createApiSessionGuard } = middleware;

function fakeRes() {
  const res = { statusCode: null, body: null, redirectedTo: null };
  res.status = code => { res.statusCode = code; return res; };
  res.json = body => { res.body = body; return res; };
  res.redirect = target => { res.redirectedTo = target; return res; };
  return res;
}

test("calls through when a session token is present", async () => {
  const guard = createApiSessionGuard(async () => "a-token");
  const res = fakeRes();
  let calledNext = false;

  await guard({}, res, () => { calledNext = true; });

  assert.equal(calledNext, true);
  assert.equal(res.statusCode, null);
});

test("answers 401 when there is no token", async () => {
  const guard = createApiSessionGuard(async () => null);
  const res = fakeRes();
  let calledNext = false;

  await guard({}, res, () => { calledNext = true; });

  assert.equal(calledNext, false);
  assert.equal(res.statusCode, 401);
});

test("401 carries a JSON body and never a redirect", async () => {
  const guard = createApiSessionGuard(async () => null);
  const res = fakeRes();

  await guard({}, res, () => {});

  assert.equal(res.redirectedTo, null);
  assert.equal(typeof res.body, "object");
  assert.match(res.body.error, /sign in/i);
});

test("an empty string token counts as no session", async () => {
  const guard = createApiSessionGuard(async () => "");
  const res = fakeRes();

  await guard({}, res, () => {});

  assert.equal(res.statusCode, 401);
});

test("a Redis failure is a 503, not a 401", async () => {
  const guard = createApiSessionGuard(async () => {
    throw new Error("Redis unreachable");
  });
  const res = fakeRes();
  let calledNext = false;

  await guard({}, res, () => { calledNext = true; });

  // telling someone to sign in again would not help and would lose their work
  assert.equal(res.statusCode, 503);
  assert.equal(calledNext, false);
  assert.equal(res.redirectedTo, null);
});

test("the guard never leaks the underlying error to the caller", async () => {
  const guard = createApiSessionGuard(async () => {
    throw new Error("connect ECONNREFUSED 10.0.0.5:6379");
  });
  const res = fakeRes();

  await guard({}, res, () => {});

  assert.doesNotMatch(res.body.error, /ECONNREFUSED/);
  assert.doesNotMatch(res.body.error, /6379/);
});
