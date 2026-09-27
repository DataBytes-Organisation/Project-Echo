export const ROLES = Object.freeze([
  "reviewer-1",
  "reviewer-2",
  "adjudicator",
]);

const ACTIONABLE_STATUS_BY_ROLE = Object.freeze({
  "reviewer-1": "awaiting_first_review",
  "reviewer-2": "awaiting_second_review",
  adjudicator: "awaiting_adjudication",
});

function assertRole(role) {
  if (!ROLES.includes(role)) {
    throw new TypeError("Choose a supported review role.");
  }
}

export function isActionableForRole(reviewStatus, role) {
  assertRole(role);
  return reviewStatus === ACTIONABLE_STATUS_BY_ROLE[role];
}

function displayRecord(record, sessions, role) {
  const reviewStatus = sessions.get(record.id)?.status ?? null;
  return Object.freeze({
    ...record,
    reviewStatus,
    isActionable: isActionableForRole(reviewStatus, role),
  });
}

export function orderRecordsForRole(records, sessions, role) {
  assertRole(role);
  const actionable = [];
  const notActionable = [];

  for (const record of records) {
    const display = displayRecord(record, sessions, role);
    (display.isActionable ? actionable : notActionable).push(display);
  }

  return Object.freeze([...actionable, ...notActionable]);
}

export function firstActionableDetectionId(records, sessions, role) {
  assertRole(role);

  for (const record of records) {
    if (isActionableForRole(sessions.get(record.id)?.status ?? null, role)) {
      return record.id;
    }
  }

  return null;
}
