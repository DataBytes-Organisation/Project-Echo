import {
  DetectionValidationError,
  validateDetectionRecord,
} from "./detection-record.mjs";

export const INVALID_FIXTURE_MESSAGE = "Detection records could not be loaded because the fixture data is invalid.";

export class DetectionRepositoryError extends Error {
  constructor(message, options) {
    super(message, options);
    this.name = "DetectionRepositoryError";
    this.userMessage = message;
  }
}

export class DetectionReviewRepository {
  async list() {
    throw new Error("list() must be implemented");
  }

  async load(_id) {
    throw new Error("load() must be implemented");
  }
}

function validateFixture(fixture) {
  try {
    return validateDetectionRecord(fixture);
  } catch (error) {
    if (error instanceof DetectionValidationError) {
      throw new DetectionRepositoryError(INVALID_FIXTURE_MESSAGE, { cause: error });
    }

    throw error;
  }
}

export class FixtureDetectionReviewRepository extends DetectionReviewRepository {
  constructor(fixtures) {
    super();
    this.fixtures = [...fixtures];
  }

  async list() {
    return this.fixtures.map(validateFixture);
  }

  async load(id) {
    const fixture = this.fixtures.find(item => item?.id === id);
    return fixture ? validateFixture(fixture) : null;
  }
}
