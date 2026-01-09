# Project Echo UI Style Guide (v0.2) — Admin (updated Dec 2025)

This file contains the Project Echo Admin Dashboard style guide (v0.2) with the additional CSS/implementation notes and snippets used in the repo. It includes the sidebar hover/focus updates and examples for integrating the guide's variables into your global stylesheet.

Summary
- Primary action color: `--primary: #2F6E4F` (green)
- Accent: `--accent: #DDECE3` (soft green)
- Page background: `--page: #F5F1EA`
- Surface background: `--bg: #FFFFFF`

1) Principles
- Simple & tough: minimal decoration, high contrast, few variants.
- 8‑pt spacing grid.

2) Theme Tokens (wireframe skin)
```css
:root{
  --page:#F5F1EA; --bg:#FFFFFF;
  --ink:#1F2D23; --muted:#6F7B74; --border:#E4E1D9;
  --primary:#2F6E4F; --accent:#DDECE3;
  --success:#2F6E4F; --warning:#D29B38; --danger:#C8473C;
  --radius-window:12px; --radius:8px;
  --space-1:8px; --space-2:16px; --space-3:24px;
  --shadow-card:0 4px 12px rgba(0,0,0,.08);
  --shadow-soft:0 1px 2px rgba(0,0,0,.06);
  --font-sans: Inter, system-ui, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
```

3) Sidebar (Left Toolbar) — Implementation notes
The Admin sidebar should follow the light spec. Key points implemented in the repo:
- Width: 240px (collapsed: 64px). Background: solid `var(--page)`.
- Item height: 40px; left padding: 16px; gap: 8px; icon size: 20px.
- Text: 14px Medium; color `var(--ink)` (inactive 80% via rgba if needed).
- Active state: 3px left bar in `var(--primary)`, item bg `#EAF3EC`.
- Hover: light green `#F0F7F2` (consistent with sample CSS below).

4) Accessibility
- Text contrast ≥ AA.
- Visible focus on interactive elements: 2px ring in `--primary`.
- Hit target ≥ 40×40 px for icons/controls.

5) Inline CSS snippets used in the components
Use these snippets as a starting point; ideally move them into your global stylesheet (e.g., `admin.css`) and prefer variables above hard-coded values.

Sidebar hover / focus / active (recommended)
```css
/* Sidebar base */
.sidebar, .left-sidebar{width:240px;background:var(--page);border-right:1px solid var(--border);}
.sidebar ul{list-style:none;margin:0;padding:0}
.sidebar a{display:block;padding:0 var(--space-2);height:40px;line-height:40px;color:var(--ink);font:500 14px var(--font-sans);text-decoration:none;transition:background .12s ease,color .12s ease}

/* Hover: light green */
.sidebar a:hover{background:#F0F7F2;color:var(--ink)}

/* Focus ring: visible and accessible */
.sidebar a:focus{outline:2px solid rgba(47,110,79,0.12);outline-offset:2px}

/* Active state: left bar + background */
.sidebar .active{background:#EAF3EC;position:relative}
.sidebar .active::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--primary)}

/* Icon colour rule (use CSS instead of inline styles) */
.sidebar .icon i{color:var(--primary);width:20px;height:20px;display:inline-block}

/* CTA button styling (full width) */
.sidebar .cta{display:block;text-align:center;background:var(--primary);color:#fff;padding:10px 12px;border-radius:8px;font-weight:600;text-decoration:none}
.sidebar .cta:hover{filter:brightness(.96)}
```

6) Migration notes — what I changed in the repo
- Replaced the previous icon-heavy sidebar HTML with a minimal text-link sidebar in `sidebar-component.html` and added `sidebar-component2.html` as an alternate copy.
- Added inline `<style>` blocks in both sidebar files to ensure the hover (#F0F7F2) and focus styles are immediately applied. These should be moved into the global admin stylesheet.

7) Recommended next steps (apply globally)
1. Add the variables (the `:root` block above) into your global CSS (e.g., `src/styles/admin.css` or similar).
2. Move the sidebar rules from this file into the global stylesheet and remove inline `<style>` blocks from components.
3. Add `.active` class logic in your JS router or server-rendering to mark the current page's sidebar item as active.
4. Use `aria-current="page"` on the active link for accessibility where appropriate.

8) Example: replace inline icon colour styles
Search for all occurrences of `style="color: #2F6E4F;"` in components and replace with CSS targeting `.sidebar .icon i` to keep styling consistent and centralized.

9) Notes on collapsing behavior
- Keep `.hide-menu` spans for text; CSS can hide/show depending on `.collapsed` state.
- Collapsed width: set `.sidebar.collapsed{width:64px}` and hide text labels with `.sidebar.collapsed .hide-menu{display:none}`.

10) Contact
If you want, I can move these snippets into your project's main stylesheet and remove the inline styles now present in the components.

---
File created/updated: `src/Components/HMI/ui/public/admin/component/style_sheet.md`
