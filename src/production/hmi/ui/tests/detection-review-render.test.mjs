import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const render = require("../public/admin/js/detection-review-render.js");

// The Backend accepts and returns markup in a detection's species field, so
// stored values must be rendered as text, never as HTML.
const MARKUP_SPECIES = "<img src=x onerror=alert(1)>";
const ATTRIBUTE_BREAKOUT = '"><script>alert(1)</script>';
const DANGEROUS_TAGS = /<(img|script|svg|iframe|style|object)\b/i;

function detection(overrides = {}) {
  return {
    _id: "651f2a9f4d1f1b1c3e2a4567",
    species: "Litoria inermis",
    timestamp: "2026-09-15T12:12:02Z",
    confidence: 1.6,
    ...overrides,
  };
}

test("list item renders markup in the species name as plain text", () => {
  const html = render.detectionListItemHtml(detection({ species: MARKUP_SPECIES }));

  assert.doesNotMatch(html, DANGEROUS_TAGS);
  assert.ok(html.includes("&lt;img src=x onerror=alert(1)&gt;"));
});

test("list item cannot be broken out of its data-id attribute", () => {
  const html = render.detectionListItemHtml(detection({ _id: ATTRIBUTE_BREAKOUT }));

  assert.doesNotMatch(html, DANGEROUS_TAGS);
  assert.ok(html.includes('data-id="&quot;&gt;&lt;script&gt;alert(1)&lt;/script&gt;"'));
});

test("detail heading renders markup in the species name as plain text", () => {
  const html = render.detailHtml({
    detection: detection({ species: MARKUP_SPECIES }),
    similar: { ambiguous: false, novel: false },
    queryAudioUrl: "blob:http://localhost/abc",
    rows: [],
  });

  assert.doesNotMatch(html, DANGEROUS_TAGS);
  assert.ok(html.includes("<h5>Reviewing: &lt;img src=x onerror=alert(1)&gt;</h5>"));
});

test("similar-result rows render markup in the species name as plain text", () => {
  const html = render.detailHtml({
    detection: detection(),
    similar: { ambiguous: true, novel: false },
    queryAudioUrl: "blob:http://localhost/abc",
    rows: [
      { species: MARKUP_SPECIES, similarity: 0.9, isMatchOfTop: false, audioUrl: "blob:http://localhost/def" },
      { species: ATTRIBUTE_BREAKOUT, similarity: 0.3, isMatchOfTop: false, audioUrl: "blob:http://localhost/ghi" },
    ],
  });

  assert.doesNotMatch(html, DANGEROUS_TAGS);
  assert.ok(html.includes("&lt;img src=x onerror=alert(1)&gt;"));
  assert.ok(html.includes("&quot;&gt;&lt;script&gt;alert(1)&lt;/script&gt;"));
});

test("an audio URL cannot break out of its src attribute", () => {
  const html = render.similarRowHtml({
    species: "Litoria inermis",
    similarity: 0.9,
    isMatchOfTop: true,
    audioUrl: '" onerror="alert(1)',
  });

  assert.ok(html.includes('src="&quot; onerror=&quot;alert(1)"'));
});

test("ordinary species names and scores render unchanged", () => {
  const html = render.detailHtml({
    detection: detection(),
    similar: { ambiguous: false, novel: false },
    queryAudioUrl: "blob:http://localhost/abc",
    rows: [{ species: "Litoria inermis", similarity: 0.895, isMatchOfTop: true, audioUrl: "blob:http://localhost/def" }],
  });

  assert.ok(html.includes("Reviewing: Litoria inermis"));
  assert.ok(html.includes("0.895"));
  assert.ok(html.includes("confidence 1.6%"));
});

test("non-numeric confidence and similarity render as zero instead of NaN", () => {
  const item = render.detectionListItemHtml(detection({ confidence: "not-a-number" }));
  const row = render.similarRowHtml({ species: "A", similarity: undefined, isMatchOfTop: false, audioUrl: "x" });

  assert.ok(item.includes("0.0%"));
  assert.ok(row.includes("0.000"));
  assert.ok(!row.includes("NaN"));
});

test("rows below the weak-match threshold are dimmed and labelled, strong ones are not", () => {
  const weak = render.similarRowHtml({ species: "A", similarity: 0.5, isMatchOfTop: false, audioUrl: "x" });
  const strong = render.similarRowHtml({ species: "A", similarity: 0.9, isMatchOfTop: true, audioUrl: "x" });

  assert.ok(weak.includes("dr-row-weak"));
  assert.ok(weak.includes("weak, not a strong precedent"));
  assert.ok(!strong.includes("dr-row-weak"));
  assert.ok(!strong.includes("weak, not a strong precedent"));
});

test("badges reflect the ambiguous, novel, and consistent cases", () => {
  assert.ok(render.badgesHtml({ ambiguous: true, novel: false }).includes("Ambiguous"));
  assert.ok(render.badgesHtml({ ambiguous: false, novel: true }).includes("Novel"));
  assert.ok(render.badgesHtml({ ambiguous: false, novel: false }).includes("Consistent match"));
});

test("detail shows an empty state when there is nothing to compare against", () => {
  const html = render.detailHtml({
    detection: detection(),
    similar: { ambiguous: false, novel: true },
    queryAudioUrl: "blob:http://localhost/abc",
    rows: [],
  });

  assert.ok(html.includes("No other detections with a stored embedding to compare against yet."));
});
