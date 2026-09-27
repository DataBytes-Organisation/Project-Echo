function result(value) {
  return Object.freeze(value);
}

export async function resetPrototype({
  isPending,
  confirmReset,
  reviewRepository,
  draftStore,
  preferenceStore,
}) {
  if (isPending) {
    return result({
      status: "blocked",
      message: "Wait for the current save to finish.",
    });
  }

  try {
    if (!confirmReset()) {
      return result({ status: "cancelled", message: null });
    }

    draftStore.discardAll();
    preferenceStore.reset();
    await reviewRepository.reset();
    return result({
      status: "reset",
      role: "reviewer-1",
      message: "Prototype data reset.",
    });
  } catch (_error) {
    return result({
      status: "failed",
      message: "Prototype data could not be reset.",
    });
  }
}
