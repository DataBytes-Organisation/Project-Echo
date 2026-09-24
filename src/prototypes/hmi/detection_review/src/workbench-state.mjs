export function createInitialReviewState() {
  return Object.freeze({
    status: "loading",
    records: [],
    selectedId: null,
    errorMessage: null,
  });
}

const FALLBACK_ERROR_MESSAGE = "Detection records could not be loaded. Check the fixture data and reload.";

function emptyState() {
  return Object.freeze({
    status: "empty",
    records: [],
    selectedId: null,
    errorMessage: null,
  });
}

function populatedState(records) {
  return Object.freeze({
    status: "populated",
    records,
    selectedId: records[0].id,
    errorMessage: null,
  });
}

function failedState(error) {
  return Object.freeze({
    status: "failed",
    records: [],
    selectedId: null,
    errorMessage: typeof error?.userMessage === "string"
      ? error.userMessage
      : FALLBACK_ERROR_MESSAGE,
  });
}

export function createReviewWorkbench(repository, onChange = () => {}) {
  let state = createInitialReviewState();

  function setState(nextState) {
    state = nextState;
    onChange(state);
  }

  return {
    getState: () => state,
    async load() {
      setState(createInitialReviewState());

      try {
        const records = await repository.list();
        setState(records.length === 0 ? emptyState() : populatedState(records));
      } catch (error) {
        setState(failedState(error));
      }

      return state;
    },
    select(id) {
      if (state.status === "populated"
        && state.records.some(record => record.id === id)) {
        setState(Object.freeze({ ...state, selectedId: id }));
      }

      return state;
    },
  };
}
