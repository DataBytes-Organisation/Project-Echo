# HMI (Human-Machine Interface)

Owned by the frontend/HMI team. Web dashboard for viewing live detections,
sensor status, and species insights.

- `ui/` — Node/Express + frontend app (see `ui/package.json` for scripts).
- `ai/` — bio/species reference spreadsheets used by the dashboard.
- `digital_assets/`, `recodings/` — media assets used by the UI.
- `HMI.Dockerfile` — container build, served on port 3000 (see `docs/PORTS.md`).
