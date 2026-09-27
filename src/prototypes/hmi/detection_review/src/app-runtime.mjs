import { createAsyncViewGuard } from "./app-coordination.mjs";
import { resetPrototype } from "./prototype-reset.mjs";

function result(value) {
  return Object.freeze(value);
}

export function beginRecognizedWorkflowSubmission(
  event,
  runtime,
  { canSubmit = true } = {},
) {
  event.preventDefault();

  if (!canSubmit) {
    return result({ status: "blocked", token: null });
  }

  return runtime.beginSubmission();
}

export function createAppRuntimeController({
  initialRole,
  getDetectionId,
  preferenceStore,
}) {
  let activeRole = initialRole;
  let isPending = false;

  function identity() {
    return {
      detectionId: getDetectionId(),
      actor: activeRole,
    };
  }

  const viewGuard = createAsyncViewGuard(identity);

  return Object.freeze({
    get activeRole() {
      return activeRole;
    },

    get isPending() {
      return isPending;
    },

    beginView() {
      return viewGuard.begin(identity());
    },

    isCurrent(token) {
      return viewGuard.isCurrent(token);
    },

    beginNavigation(navigate) {
      if (isPending) {
        return result({ status: "blocked", token: null });
      }

      navigate();
      return result({ status: "started", token: viewGuard.begin(identity()) });
    },

    switchRole(nextRole) {
      if (isPending) {
        return result({ status: "blocked", role: activeRole });
      }

      viewGuard.begin(identity());
      preferenceStore.saveRole(nextRole);
      activeRole = nextRole;
      return result({ status: "switched", role: activeRole });
    },

    beginSubmission() {
      if (isPending) {
        return result({ status: "blocked", token: null });
      }

      isPending = true;
      return result({ status: "started", token: viewGuard.capture(identity()) });
    },

    completeSuccessfulSubmission(token, cleanup) {
      let cleanupResult;
      try {
        cleanupResult = cleanup();
      } finally {
        isPending = false;
      }

      return result({
        status: viewGuard.isCurrent(token) ? "current" : "stale",
        cleanupResult,
      });
    },

    completeFailedSubmission(token) {
      isPending = false;
      return result({
        status: viewGuard.isCurrent(token) ? "current" : "stale",
      });
    },

    async reset(options) {
      const resetResult = await resetPrototype({
        ...options,
        isPending,
      });

      if (resetResult.status === "reset") {
        viewGuard.begin(identity());
        activeRole = resetResult.role;
      }

      return resetResult;
    },
  });
}
