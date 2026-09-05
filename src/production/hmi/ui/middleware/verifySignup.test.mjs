// First-ever test for middleware/verifySignup.js's confirmPassword - a small,
// pure Express middleware (no DB) that had zero test coverage before this.
// Mirrors the backend's test_errors.py as the HMI-side "first unit test"
// example in docs/team-guides/TDD_Guide.md.
//
// Run: node --test middleware/verifySignup.test.mjs
// (or via `npm test`, once wired into package.json's test script)

import test from "node:test";
import assert from "node:assert/strict";
import verifySignUp from "./verifySignup.js";

function makeReqRes(body) {
  const req = { body };
  const res = {
    statusCode: null,
    body: null,
    status(code) {
      this.statusCode = code;
      return this;
    },
    send(payload) {
      this.body = payload;
      return this;
    },
  };
  return { req, res };
}

test("confirmPassword calls next() when password and confirmpassword match", () => {
  const { req, res } = makeReqRes({ password: "hunter2", confirmpassword: "hunter2" });
  let nextCalled = false;
  verifySignUp.confirmPassword(req, res, () => {
    nextCalled = true;
  });
  assert.equal(nextCalled, true);
  assert.equal(res.statusCode, null);
});

test("confirmPassword responds 400 and does not call next() when they differ", () => {
  const { req, res } = makeReqRes({ password: "hunter2", confirmpassword: "wrong" });
  let nextCalled = false;
  verifySignUp.confirmPassword(req, res, () => {
    nextCalled = true;
  });
  assert.equal(nextCalled, false);
  assert.equal(res.statusCode, 400);
});
