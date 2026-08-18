const DECISION_LABELS = Object.freeze({
  confirmed: "Confirmed",
  rejected: "Rejected",
  corrected_species: "Corrected species",
  insufficient_evidence: "Insufficient evidence",
});

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function decisionLabel(decision) {
  return DECISION_LABELS[decision] ?? "Unknown decision";
}

function statusLabel(status) {
  return status.replaceAll("_", " ");
}

function fieldError(field, errors) {
  return errors[field]
    ? `<p class="field-error" id="${field}-error" data-error-for="${field}">${escapeHtml(errors[field])}</p>`
    : "";
}

function decisionOptions(selectedDecision = "") {
  return Object.entries(DECISION_LABELS).map(([value, label]) => `
    <label class="decision-option">
      <input
        type="radio"
        name="decision"
        value="${value}"
        required
        ${selectedDecision === value ? "checked" : ""}
      >
      <span>
        <strong>${label}</strong>
        <small>${value === "confirmed"
    ? "The model prediction is supported by the evidence."
    : value === "rejected"
      ? "The model prediction is not supported."
      : value === "corrected_species"
        ? "The evidence supports a different species."
        : "The available evidence cannot support a decision."}</small>
      </span>
    </label>`).join("");
}

function decisionFields(values, errors, { adjudication = false } = {}) {
  const reasonName = adjudication ? "resolutionReason" : "reason";
  const reasonLabel = adjudication ? "Resolution reason" : "Decision reason";
  const reasonValue = values[reasonName] ?? "";

  return `
    <fieldset class="decision-fieldset" ${errors.decision ? "aria-describedby=\"decision-error\"" : ""}>
      <legend>${adjudication ? "Final result" : "Review decision"}</legend>
      <div class="decision-grid">${decisionOptions(values.decision)}</div>
      ${fieldError("decision", errors)}
    </fieldset>
    <div class="form-field" data-corrected-species-field ${values.decision === "corrected_species" ? "" : "hidden"}>
      <label for="corrected-species">Corrected species</label>
      <input
        id="corrected-species"
        name="correctedSpecies"
        value="${escapeHtml(values.correctedSpecies)}"
        autocomplete="off"
        ${values.decision === "corrected_species" ? "required" : ""}
        ${errors.correctedSpecies ? "aria-invalid=\"true\" aria-describedby=\"correctedSpecies-error\"" : ""}
      >
      ${fieldError("correctedSpecies", errors)}
    </div>
    <div class="form-field" data-reason-field ${adjudication || (values.decision && values.decision !== "confirmed") ? "" : "hidden"}>
      <label for="review-reason">${reasonLabel}</label>
      <textarea
        id="review-reason"
        name="${reasonName}"
        rows="4"
        ${adjudication || (values.decision && values.decision !== "confirmed") ? "required" : ""}
        ${errors[reasonName] ? `aria-invalid="true" aria-describedby="${reasonName}-error"` : ""}
      >${escapeHtml(reasonValue)}</textarea>
      <small>${adjudication
    ? "Explain how the final result resolves the disagreement."
    : "Required for rejection, correction, or insufficient evidence."}</small>
      ${fieldError(reasonName, errors)}
    </div>`;
}

function reviewForm(session, options) {
  const { errors = {}, values = {} } = options;
  return `
    <form
      class="decision-form"
      data-review-form
      data-actor="${escapeHtml(session.actor)}"
      aria-describedby="review-form-guidance"
      novalidate
    >
      <p class="form-guidance" id="review-form-guidance">Choose one decision. Required supporting fields appear after your choice.</p>
      ${decisionFields(values, errors)}
      ${options.submissionError
    ? `<p class="submission-error" role="alert">${escapeHtml(options.submissionError)}</p>`
    : ""}
      <button class="primary-button" type="submit">Submit independent review</button>
      <p class="submission-note">This stateful action is attempted once. A failed save is never retried automatically.</p>
    </form>`;
}

function adjudicationForm(session, options) {
  const { errors = {}, values = {} } = options;
  return `
    <form
      class="decision-form"
      data-adjudication-form
      data-actor="${escapeHtml(session.actor)}"
      aria-describedby="adjudication-form-guidance"
      novalidate
    >
      <p class="form-guidance" id="adjudication-form-guidance">Choose the final result and explain how it resolves the disagreement.</p>
      ${decisionFields(values, errors, { adjudication: true })}
      ${options.submissionError
    ? `<p class="submission-error" role="alert">${escapeHtml(options.submissionError)}</p>`
    : ""}
      <button class="primary-button" type="submit">Finalize adjudication</button>
      <p class="submission-note">Finalization is attempted once. A failed save is never retried automatically.</p>
    </form>`;
}

function submissionCard(title, submission) {
  return `
    <article class="submission-card">
      <p>${escapeHtml(title)}</p>
      <strong>${escapeHtml(decisionLabel(submission.decision))}</strong>
      ${submission.correctedSpecies
    ? `<span>Species: ${escapeHtml(submission.correctedSpecies)}</span>`
    : ""}
      ${submission.reason
    ? `<blockquote>${escapeHtml(submission.reason)}</blockquote>`
    : ""}
    </article>`;
}

function independentReviewSummary(session) {
  const first = session.submissions["reviewer-1"];
  const second = session.submissions["reviewer-2"];

  if (!first || !second) {
    return "";
  }

  return `
    <div class="submission-comparison" aria-label="Independent review comparison">
      ${submissionCard("Reviewer 1", first)}
      ${submissionCard("Reviewer 2", second)}
    </div>`;
}

function renderHistory(history, headingId = "case-history-heading") {
  const entries = history.map(entry => `
    <li>
      <strong>${escapeHtml(entry.action.replaceAll("_", " "))}</strong>
      <span>${escapeHtml(entry.actor)} · ${escapeHtml(entry.previousStatus)} → ${escapeHtml(entry.resultingStatus)}</span>
      <time datetime="${escapeHtml(entry.timestamp)}">${escapeHtml(entry.timestamp)}</time>
    </li>`).join("");

  return `
    <section class="history-panel" aria-labelledby="${headingId}">
      <div class="history-panel__heading">
        <h3 id="${headingId}">Case audit history</h3>
        <span>${history.length} ${history.length === 1 ? "entry" : "entries"}</span>
      </div>
      <p class="history-panel__description">Append-only record of case actions and status transitions.</p>
      ${history.length === 0
    ? '<p class="history-empty">No case actions have been recorded yet.</p>'
    : `<ol>${entries}</ol>`}
    </section>`;
}

export function renderDraftRecovery(recoveryState) {
  if (!recoveryState || ["idle", "discarded"].includes(recoveryState.status)) {
    return "";
  }

  if (recoveryState.status === "failed") {
    return `
      <aside class="draft-recovery draft-recovery--failed" role="alert">
        <h3>Draft recovery unavailable</h3>
        <p>${escapeHtml(recoveryState.errorMessage)}</p>
      </aside>`;
  }

  if (recoveryState.status === "restored") {
    return `
      <aside class="draft-recovery draft-recovery--restored" role="status" aria-live="polite">
        <div>
          <h3>Draft restored</h3>
          <p>Your saved input is back in the form. Review it before submitting.</p>
        </div>
        <button class="secondary-button" type="button" data-draft-discard>Discard restored draft</button>
      </aside>`;
  }

  return `
    <aside class="draft-recovery" role="status" aria-labelledby="draft-recovery-heading">
      <div>
        <h3 id="draft-recovery-heading">Recovered draft available</h3>
        <p>Saved ${escapeHtml(recoveryState.draft.savedAt)} from case version ${escapeHtml(recoveryState.draft.version)}.</p>
      </div>
      <div class="recovery-actions" aria-label="Recovered draft actions">
        <button class="primary-button" type="button" data-draft-restore>Restore draft</button>
        <button class="secondary-button" type="button" data-draft-discard>Discard draft</button>
      </div>
    </aside>`;
}

function attemptedReviewSummary(values) {
  const reason = values.resolutionReason || values.reason;
  return `
    <dl class="conflict-summary">
      <div>
        <dt>Decision</dt>
        <dd>${escapeHtml(decisionLabel(values.decision))}</dd>
      </div>
      ${values.correctedSpecies ? `
        <div>
          <dt>Corrected species</dt>
          <dd>${escapeHtml(values.correctedSpecies)}</dd>
        </div>` : ""}
      ${reason ? `
        <div>
          <dt>Entered reason</dt>
          <dd>${escapeHtml(reason)}</dd>
        </div>` : ""}
    </dl>`;
}

export function renderConflictRecovery(conflict) {
  const latest = conflict.latestSession;
  return `
    <section class="panel workflow-panel conflict-panel" aria-labelledby="conflict-heading" data-workflow-status="conflict">
      <header class="panel-heading">
        <div>
          <h2 id="conflict-heading" tabindex="-1">Review changed before save</h2>
          <p role="alert">Your entered data is preserved. Nothing was submitted again.</p>
        </div>
        <span class="workflow-status">conflict</span>
      </header>
      <div class="workflow-body">
        <div class="conflict-comparison" aria-label="Stale write comparison">
          <article class="conflict-version">
            <h3>Your unsaved review</h3>
            <p class="version-label">Version ${escapeHtml(conflict.expectedVersion)}</p>
            ${attemptedReviewSummary(conflict.attemptedValues)}
          </article>
          <article class="conflict-version">
            <h3>Latest case</h3>
            <p class="version-label">Version ${escapeHtml(latest.version)}</p>
            <dl class="conflict-summary">
              <div>
                <dt>Current status</dt>
                <dd>${escapeHtml(statusLabel(latest.status))}</dd>
              </div>
            </dl>
          </article>
        </div>
        <p class="submission-note">Keeping the draft updates its version reference only. You must review and submit it again yourself.</p>
        <div class="recovery-actions" aria-label="Conflict recovery actions">
          <button class="primary-button" type="button" data-conflict-keep-draft>Keep draft with latest case</button>
          <button class="secondary-button" type="button" data-conflict-discard-draft>Discard draft and use latest</button>
        </div>
        ${renderHistory(latest.history, "conflict-history-heading")}
      </div>
    </section>`;
}

function workflowBody(session, options) {
  if (session.status === "awaiting_first_review") {
    return `
      <div class="workflow-intro">
        <p class="eyebrow">Reviewer 1 session</p>
        <h3>First independent review</h3>
        <p>Record an evidence-led decision. Reviewer 2 will not see it before submitting independently.</p>
      </div>
      ${reviewForm(session, options)}`;
  }

  if (session.status === "awaiting_second_review") {
    if (session.actor !== "reviewer-2") {
      return `
        <div class="workflow-result" role="status">
          <p class="eyebrow">Review recorded</p>
          <h3>Waiting for reviewer 2</h3>
          <p>Your first review is stored. Its decision remains hidden from the second reviewer.</p>
        </div>`;
    }

    return `
      <div class="blind-notice" role="status">
        <strong>Blind review active</strong>
        <span>The first review is complete. Its decision and notes remain hidden until you submit.</span>
      </div>
      <div class="workflow-intro">
        <p class="eyebrow">Reviewer 2 session</p>
        <h3>Second independent review</h3>
        <p>Assess the same evidence without influence from reviewer 1’s decision.</p>
      </div>
      ${reviewForm(session, options)}`;
  }

  if (session.status === "consensus") {
    return `
      <div class="workflow-result workflow-result--success" role="status">
        <p class="eyebrow">Automatic comparison complete</p>
        <h3>Consensus reached</h3>
        <p>Both independent reviewers selected <strong>${escapeHtml(decisionLabel(session.consensus.decision))}</strong>${session.consensus.correctedSpecies
    ? ` for <strong>${escapeHtml(session.consensus.correctedSpecies)}</strong>`
    : ""}.</p>
      </div>
      ${independentReviewSummary(session)}`;
  }

  if (session.status === "awaiting_adjudication") {
    return `
      <div class="workflow-result workflow-result--warning" role="status">
        <p class="eyebrow">Independent decisions differ</p>
        <h3>Adjudication required</h3>
        <p>Compare both submissions, then record one final result and a resolution reason.</p>
      </div>
      ${independentReviewSummary(session)}
      ${session.actor === "adjudicator" ? adjudicationForm(session, options) : ""}`;
  }

  if (session.status === "finalized") {
    return `
      <div class="workflow-result workflow-result--success" role="status">
        <p class="eyebrow">Case closed</p>
        <h3>Adjudication finalized</h3>
        <p>Final result: <strong>${escapeHtml(decisionLabel(session.adjudication.decision))}</strong>${session.adjudication.correctedSpecies
    ? ` for <strong>${escapeHtml(session.adjudication.correctedSpecies)}</strong>`
    : ""}.</p>
        <blockquote>${escapeHtml(session.adjudication.resolutionReason)}</blockquote>
      </div>
      ${independentReviewSummary(session)}`;
  }

  return `<p class="submission-error" role="alert">This review state cannot be displayed.</p>`;
}

export function renderReviewWorkflow(session, options = {}) {
  const recoveryValues = options.recoveryState?.status === "restored"
    ? options.recoveryState.values
    : null;
  const resolvedOptions = {
    ...options,
    values: options.values ?? recoveryValues ?? {},
  };

  return `
    <section class="panel workflow-panel" aria-labelledby="workflow-heading" data-workflow-status="${escapeHtml(session.status)}">
      <header class="panel-heading">
        <div>
          <h2 id="workflow-heading">Review workflow</h2>
          <p>Independent verification, consensus, and adjudication.</p>
        </div>
        <span class="workflow-status">${escapeHtml(statusLabel(session.status))}</span>
      </header>
      <div class="workflow-body">
        ${renderDraftRecovery(options.recoveryState)}
        ${workflowBody(session, resolvedOptions)}
        ${renderHistory(session.history)}
      </div>
    </section>`;
}

export function renderReviewWorkflowFailure(message) {
  return `
    <section class="panel workflow-panel" aria-labelledby="workflow-heading" data-workflow-status="failed">
      <header class="panel-heading">
        <div>
          <h2 id="workflow-heading">Review workflow</h2>
          <p>Independent verification, consensus, and adjudication.</p>
        </div>
        <span class="workflow-status">unavailable</span>
      </header>
      <div class="workflow-body">
        <div class="workflow-result workflow-result--warning" role="alert">
          <p class="eyebrow">Workflow unavailable</p>
          <h3>Review state could not be loaded</h3>
          <p>${escapeHtml(message)}</p>
        </div>
        <button class="primary-button" type="button" data-workflow-retry>Try again manually</button>
      </div>
    </section>`;
}

export function renderAdjudicationQueue(sessions) {
  const countLabel = `${sessions.length} ${sessions.length === 1 ? "case" : "cases"}`;
  const items = sessions.length === 0
    ? `<p class="adjudication-empty">No disagreements are waiting for adjudication.</p>`
    : `<ol>${sessions.map(session => `
        <li>
          <a href="?scenario=adjudication&detection=${encodeURIComponent(session.detectionId)}">
            <strong>${escapeHtml(session.detectionId)}</strong>
            <span>Compare two independent decisions</span>
          </a>
        </li>`).join("")}</ol>`;

  return `
    <aside class="adjudication-queue" aria-labelledby="adjudication-queue-heading">
      <div>
        <p class="eyebrow">Adjudicator session</p>
        <h2 id="adjudication-queue-heading">Disagreement queue</h2>
      </div>
      <span class="record-count">${countLabel}</span>
      ${items}
    </aside>`;
}
