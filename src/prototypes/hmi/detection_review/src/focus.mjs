export function restoreQueueItemFocus(root, detectionId) {
  const selectedItem = [...root.querySelectorAll("[data-detection-id]")]
    .find(item => item.dataset.detectionId === detectionId);

  if (!selectedItem || typeof selectedItem.focus !== "function") {
    return false;
  }

  selectedItem.focus({ preventScroll: true });
  return true;
}
