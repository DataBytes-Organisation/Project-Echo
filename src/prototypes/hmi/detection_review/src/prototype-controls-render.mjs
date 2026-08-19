import { ROLES } from "./role-workspace.mjs";

const ROLE_LABELS = Object.freeze({
  "reviewer-1": "Reviewer 1",
  "reviewer-2": "Reviewer 2",
  adjudicator: "Adjudicator",
});

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

export function renderPrototypeControls({
  activeRole = "reviewer-1",
  isPending = false,
  resetStatus = null,
}) {
  const disabled = isPending ? "disabled" : "";
  const roleButtons = ROLES.map(role => `
    <button
      type="button"
      class="role-switcher__button"
      data-role="${role}"
      aria-pressed="${role === activeRole}"
      ${disabled}
    >${ROLE_LABELS[role]}</button>`).join("");
  const resetMessage = resetStatus?.message
    ? `<p class="prototype-controls__status" role="status" aria-live="polite">${escapeHtml(resetStatus.message)}</p>`
    : "";

  return `
    <section class="prototype-controls" aria-label="Prototype controls">
      <div class="prototype-controls__roles">
        <div class="role-switcher" role="group" aria-label="Active review role">
          ${roleButtons}
        </div>
      </div>
      <div class="prototype-controls__reset">
        <div class="prototype-reset">
          <button type="button" class="secondary-button" data-prototype-reset ${disabled}>Reset prototype data</button>
        </div>
      </div>
      ${resetMessage}
    </section>`;
}
