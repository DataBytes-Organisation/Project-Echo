import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const uiDir = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const repoDir = resolve(uiDir, "../../../..");
const manifest = JSON.parse(readFileSync(resolve(uiDir, "package.json"), "utf8"));

test("generated dependencies and the stray lockfile are not tracked", () => {
  const tracked = execFileSync("git", ["ls-files", "src/production/hmi"], {
    cwd: repoDir,
    encoding: "utf8",
  }).trim().split(/\r?\n/).filter(Boolean);

  assert.equal(tracked.some((path) => path.includes("/node_modules/")), false);
  assert.equal(tracked.includes("src/production/hmi/package-lock.json"), false);
});

test("runtime dependencies match direct server imports", () => {
  assert.equal(manifest.dependencies.mongodb, "^4.16.0");
  assert.equal(manifest.dependencies.nodemon, undefined);
  assert.equal(manifest.devDependencies.nodemon, "^2.0.22");

  for (const name of [
    "apexcharts", "bcryptjs", "email-validator", "express-mongo-sanitize",
    "jquery", "mqtt", "ol", "path", "requirejs", "serve-index", "simplebar",
  ]) {
    assert.equal(manifest.dependencies[name], undefined, `${name} should not be a runtime dependency`);
  }
});

test("the deployable package owns the only HMI lockfile", () => {
  assert.equal(existsSync(resolve(uiDir, "package-lock.json")), true);
  assert.equal(existsSync(resolve(uiDir, "../package-lock.json")), false);
});

test("HMI setup docs use the reproducible install and current port", () => {
  const readme = readFileSync(resolve(uiDir, "README.md"), "utf8");

  assert.match(readme, /npm ci/);
  assert.match(readme, /npm test/);
  assert.match(readme, /localhost:3000/);
  assert.doesNotMatch(readme, /localhost:8080|ui\/node_modules|npm install/);
});

test("HMI environment docs use port 3000", () => {
  const envExample = readFileSync(resolve(uiDir, ".env.example"), "utf8");
  const readme = readFileSync(resolve(uiDir, "README.md"), "utf8");

  assert.match(envExample, /localhost:3000/);
  assert.match(readme, /COOKIE_SECRET/);
  assert.match(readme, /localhost:3000/);
  assert.doesNotMatch(envExample, /localhost:8080/);
  assert.doesNotMatch(readme, /localhost:8080/);
  assert.equal(existsSync(resolve(uiDir, "ENV_SETUP.md")), false, "ENV_SETUP.md was merged into README.md");
});
