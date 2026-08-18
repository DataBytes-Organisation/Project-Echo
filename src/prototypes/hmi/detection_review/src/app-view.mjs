import { renderWorkbench } from "./render.mjs";
import {
  renderAdjudicationQueue,
  renderConflictRecovery,
  renderReviewWorkflow,
  renderReviewWorkflowFailure,
} from "./workflow-render.mjs";

export function renderApplicationView({
  detectionState,
  workflowActor,
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

  return `
    <div class="application-view" data-page-state="${pageState}">
      <p class="sr-only" role="status" aria-live="polite">Current page state: ${pageState.replaceAll("-", " ")}.</p>
      ${renderWorkbench(detectionState, {
    workflowHtml,
    adjudicationQueueHtml,
  })}
    </div>`;
}
