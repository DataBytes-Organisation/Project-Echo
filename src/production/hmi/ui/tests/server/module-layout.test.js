const assert = require("node:assert/strict");
const test = require("node:test");

test("server modules load from the unified server tree", () => {
  assert.equal(typeof require("../../server/middleware").checkUserSession, "function");
  assert.equal(typeof require("../../server/routes/sensor.routes"), "function");
  assert.equal(typeof require("../../server/routes/razorpay.routes").createOrder, "function");
  assert.equal(typeof require("../../server/services/apiClient").get, "function");
  assert.equal(typeof require("../../server/services/notifications").buildFeed, "function");
});

for (const modulePath of [
  "pages.routes",
  "notifications.routes",
  "proxy.routes",
  "admin.routes",
  "contact.routes",
  "legacy-payments.routes",
]) {
  test(`${modulePath} exposes one route registrar`, () => {
    const routeModule = require(`../../server/routes/${modulePath}`);
    assert.equal(typeof routeModule.registerRoutes, "function");
  });
}
