const NAVIGATION_KEYS = new Set(["ArrowDown", "ArrowUp", "Home", "End"]);

export function getQueueNavigationTarget(records, selectedId, key) {
  if (!Array.isArray(records) || records.length === 0 || !NAVIGATION_KEYS.has(key)) {
    return null;
  }

  if (key === "Home") {
    return records[0].id;
  }

  if (key === "End") {
    return records.at(-1).id;
  }

  const selectedIndex = records.findIndex(record => record.id === selectedId);
  const currentIndex = selectedIndex >= 0 ? selectedIndex : 0;
  const offset = key === "ArrowDown" ? 1 : -1;
  const nextIndex = (currentIndex + offset + records.length) % records.length;
  return records[nextIndex].id;
}
