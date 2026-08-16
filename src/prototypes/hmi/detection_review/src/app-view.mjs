import { renderWorkbench } from "./render.mjs";
import {
  renderAdjudicationQueue,
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
}) {
  const workflowHtml = workflowLoadError
    ? renderReviewWorkflowFailure(workflowLoadError)
    : workflowSession
      ? renderReviewWorkflow(workflowSession, workflowRenderOptions)
      : "";
  const adjudicationQueueHtml = workflowActor === "adjudicator"
    && !workflowLoadError
    ? renderAdjudicationQueue(adjudicationSessions)
    : "";

  return renderWorkbench(detectionState, {
    workflowHtml,
    adjudicationQueueHtml,
  });
}
