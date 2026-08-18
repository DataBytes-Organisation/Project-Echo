import { renderWorkbench } from "./render.mjs";
import { renderPrototypeControls } from "./prototype-controls-render.mjs";
import {
  renderAdjudicationQueue,
  renderConflictRecovery,
  renderReviewWorkflow,
  renderReviewWorkflowFailure,
} from "./workflow-render.mjs";

export function renderApplicationView({
  detectionState,
  workflowActor,
  activeRole = workflowActor ?? "reviewer-1",
  isPending = false,
  resetStatus = null,
  workflowSession,
  workflowLoadError,
  adjudicationSessions = [],
  workflowRenderOptions = {},
  draftRecovery = null,
  workflowConflict = null,
}) {
  const workflowHtml = workflowLoadError
    ? renderReviewWorkflowFailure(workflowLoadError)
    : workflowConflict
      ? renderConflictRecovery(workflowConflict)
    : workflowSession
      ? renderReviewWorkflow(workflowSession, {
        ...workflowRenderOptions,
        recoveryState: draftRecovery,
      })
      : "";
  const adjudicationQueueHtml = workflowActor === "adjudicator"
    && !workflowLoadError
    ? renderAdjudicationQueue(adjudicationSessions)
    : "";

  const pageState = workflowConflict
    ? "conflict"
    : draftRecovery?.status === "restored"
      ? "draft-restored"
      : ["consensus", "finalized"].includes(workflowSession?.status)
        ? "success"
        : workflowSession?.status === "awaiting_adjudication"
          ? "adjudication"
          : workflowSession?.status === "awaiting_second_review"
            && workflowSession?.actor === "reviewer-2"
            ? "blind-second-review"
            : workflowSession?.status === "awaiting_first_review"
              ? "first-review"
              : detectionState.status;
  const hasNoActionableWork = detectionState.status === "populated"
    && detectionState.records.length > 0
    && detectionState.records.every(record => record.isActionable === false);
  const roleLabel = activeRole === "reviewer-1"
    ? "Reviewer 1"
    : activeRole === "reviewer-2"
      ? "Reviewer 2"
      : "Adjudicator";
  const noActionableWorkHtml = hasNoActionableWork
    ? `
      <aside class="no-actionable-work" role="status">
        <h2>No cases need ${roleLabel} right now</h2>
        <p>Completed and other-role cases remain visible in the queue.</p>
      </aside>`
    : "";

  return `
    <div class="application-view" data-page-state="${pageState}">
      <p class="sr-only" role="status" aria-live="polite">Current page state: ${pageState.replaceAll("-", " ")}.</p>
      ${renderPrototypeControls({ activeRole, isPending, resetStatus })}
      ${noActionableWorkHtml}
      ${renderWorkbench(detectionState, {
    workflowHtml,
    adjudicationQueueHtml,
  })}
    </div>`;
}
