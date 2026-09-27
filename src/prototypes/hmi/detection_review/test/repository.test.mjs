import test from "node:test";
import assert from "node:assert/strict";

import {
  DetectionRepositoryError,
  FixtureDetectionReviewRepository,
  INVALID_FIXTURE_MESSAGE,
} from "../src/repository.mjs";
import { makeDetection } from "../test-support/helpers.mjs";

test("lists validated fixture records", async () => {
  const repository = new FixtureDetectionReviewRepository([
    makeDetection(),
    makeDetection({ id: "det-echo-002", species: "Koala" }),
  ]);

  const records = await repository.list();

  assert.equal(records.length, 2);
  assert.deepEqual(records.map(record => record.id), ["det-echo-001", "det-echo-002"]);
  assert.equal(Object.isFrozen(records[0]), true);
});

test("loads one validated fixture by id", async () => {
  const repository = new FixtureDetectionReviewRepository([makeDetection()]);

  const record = await repository.load("det-echo-001");

  assert.equal(record.species, "Powerful Owl");
  assert.equal(Object.isFrozen(record), true);
});

test("returns null when a fixture id does not exist", async () => {
  const repository = new FixtureDetectionReviewRepository([makeDetection()]);

  assert.equal(await repository.load("det-missing"), null);
});

test("reports malformed list data with a controlled user-facing message", async () => {
  const repository = new FixtureDetectionReviewRepository([
    makeDetection({ confidence: 101 }),
  ]);

  await assert.rejects(
    repository.list(),
    error => error instanceof DetectionRepositoryError
      && error.userMessage === INVALID_FIXTURE_MESSAGE
      && !error.userMessage.includes("confidence"),
  );
});

test("reports malformed loaded data with the same controlled message", async () => {
  const repository = new FixtureDetectionReviewRepository([
    makeDetection({ animalEstLLA: ["south", 144.9631, 24] }),
  ]);

  await assert.rejects(
    repository.load("det-echo-001"),
    error => error instanceof DetectionRepositoryError
      && error.userMessage === INVALID_FIXTURE_MESSAGE,
  );
});
