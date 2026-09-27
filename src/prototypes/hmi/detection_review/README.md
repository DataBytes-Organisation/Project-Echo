# Detection review prototype

This browser prototype explores how ecologists can verify model-generated detections before they become trusted records. It uses three deterministic detection fixtures and does not call Project Echo services.

The workflow has three roles:

1. Reviewer 1 inspects the selected detection and records a decision.
2. Reviewer 2 reviews the same evidence without seeing the first decision.
3. Matching decisions reach consensus. Different decisions are sent to the Adjudicator for a final result and resolution reason.

The prototype also demonstrates role-based queues, draft recovery, stale-write conflict handling, audit history, keyboard navigation, and responsive desktop and tablet layouts. Review progress is stored in the browser's local storage. Use **Reset prototype data** to return to the initial fixtures.

## Code architecture

`index.html` provides the `#app` mount point and loads `src/app.mjs`. The application has no frontend framework. Render functions return HTML strings, and `app.mjs` writes the composed view into `#app` whenever state changes.

```text
index.html
  -> app.mjs                    application setup and DOM event handling
     -> state and workflows     queue, roles, review state, recovery
     -> repositories            fixture data and browser local storage
     -> app-view.mjs            page composition
        -> render.mjs           queue and evidence
        -> workflow-render.mjs  review, recovery, conflict, and history
        -> prototype-controls-render.mjs
```

Browser events use delegation from the `#app` element. `app.mjs` handles changes, input, keyboard navigation, clicks, and form submissions. Each handler updates the relevant state or repository, then calls the same render path again.

| Area | Main files | Responsibility |
| --- | --- | --- |
| Entry and coordination | [`app.mjs`](src/app.mjs), [`app-runtime.mjs`](src/app-runtime.mjs), [`app-coordination.mjs`](src/app-coordination.mjs) | Creates repositories and state controllers, handles browser events, prevents stale asynchronous updates, and coordinates submissions. |
| View rendering | [`app-view.mjs`](src/app-view.mjs), [`render.mjs`](src/render.mjs), [`workflow-render.mjs`](src/workflow-render.mjs), [`prototype-controls-render.mjs`](src/prototype-controls-render.mjs) | Produces the command bar, role controls, queue, evidence record, decision forms, recovery states, and audit history. |
| UI state | [`workbench-state.mjs`](src/workbench-state.mjs), [`role-workspace.mjs`](src/role-workspace.mjs), [`recovery-state.mjs`](src/recovery-state.mjs), [`keyboard.mjs`](src/keyboard.mjs), [`focus.mjs`](src/focus.mjs) | Tracks selection and page state, orders role-specific work, manages recovery choices, and calculates keyboard and focus targets. |
| Domain workflow | [`detection-record.mjs`](src/detection-record.mjs), [`review-domain.mjs`](src/review-domain.mjs), [`review-workflow.mjs`](src/review-workflow.mjs) | Validates detections, applies review transitions, compares decisions, finalizes adjudication, and exposes role-safe review sessions. |
| Data boundaries | [`fixtures.mjs`](src/fixtures.mjs), [`repository.mjs`](src/repository.mjs), [`persistent-review-repository.mjs`](src/persistent-review-repository.mjs), [`draft-store.mjs`](src/draft-store.mjs), [`prototype-preferences.mjs`](src/prototype-preferences.mjs) | Supplies deterministic detections and stores review cases, drafts, and the active role in browser local storage. |

Tests under `test/` mirror these module boundaries. Most state, domain, repository, and rendering behavior is tested without a browser DOM.

## Run locally

Requirements:

- Node.js for the automated tests
- Python 3 for the local static server

From this directory:

```powershell
npm test
npm start
```

Open <http://localhost:4173> in a browser. No package installation is required because the prototype has no external dependencies.

To try the complete workflow, submit a decision as Reviewer 1, switch to Reviewer 2 and submit a second decision, then switch to Adjudicator if the decisions differ. Refresh the page to confirm that submitted reviews and unfinished drafts persist.

## Development states

These query parameters expose deterministic states for manual checks:

- `?fixtureState=empty` shows the empty queue.
- `?fixtureState=error` shows invalid fixture handling.
- `?simulateConflict=1` makes the next save report a version conflict.
- `?simulateStorageFailure=1` shows the browser-storage failure state.

Example: <http://localhost:4173/?fixtureState=empty>

All detections, review decisions, and failures in this prototype are local test data. Production integration still requires authenticated reviewer identities, a durable review API, and server-side audit storage.
