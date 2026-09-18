import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const uiDir = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const hmiDir = resolve(uiDir, "..");
const publicDir = join(uiDir, "public");

const candidates = [
  "public/welcome.html",
  "public/filter_menu.html",
  "public/time_lapse",
  "public/project_table",
  "public/user_request_form-updated",
  "public/admin/test_1.html",
  "public/admin/test_2.html",
  "public/admin/new 1.html",
  "public/admin/component/sidebar-component2.html",
  "public/admin/component/archive/sidebar-component_old.html",
  "public/admin/add-project.html",
  "public/admin/alerts.html",
  "public/admin/reboot.html",
  "public/admin/settings.html",
  "public/admin/styles.css",
  "public/admin/style.min.css",
  "public/admin/css/styles.css",
  "public/admin/scss",
  "public/admin/component/archive",
  "public/js/map.js",
  "public/js/animal_audio.js",
  "public/js/r.js",
  "public/js/bootstrap.bundle.min.js",
  "public/js/HMI_API_onboarding_task.json",
  "public/js/Instructions.txt",
  "public/md_files/test.md",
  "public/audio/t-rex-roar.mp3",
  "routes/movement.routes.js",
];

const textExtensions = new Set([
  ".cjs", ".css", ".html", ".js", ".json", ".md", ".mjs", ".txt",
]);

const routeSourceRoots = ["server", "routes"].map((path) => join(uiDir, path));
const browserRoots = [join(publicDir, "pages"), join(publicDir, "shared"), join(publicDir, "features")];
const dynamicRoots = [publicDir];
const testRoots = [join(uiDir, "tests"), join(uiDir, "public")];
const readmeFiles = [join(hmiDir, "README.md"), join(uiDir, "README.md")];
const dockerFiles = [join(hmiDir, "HMI.Dockerfile")];

function walk(root) {
  if (!existsSync(root)) return [];
  const found = [];
  for (const entry of readdirSync(root, { withFileTypes: true })) {
    const absolute = join(root, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === "node_modules") continue;
      found.push(...walk(absolute));
    } else if (entry.isFile()) {
      found.push(absolute);
    }
  }
  return found;
}

function isTextFile(file) {
  return textExtensions.has(file.slice(file.lastIndexOf(".")));
}

function readText(file) {
  if (!existsSync(file) || !statSync(file).isFile() || !isTextFile(file)) return "";
  return readFileSync(file, "utf8").replace(/https?:\/\/[^\s"'<>]+/g, "");
}

function rel(file) {
  return relative(uiDir, file).replace(/\\/g, "/");
}

function candidateTokens(candidate) {
  const normalized = candidate.replace(/\\/g, "/");
  const withoutPublic = normalized.startsWith("public/") ? normalized.slice("public".length) : normalized;
  const tokens = new Set([normalized, withoutPublic, `/${withoutPublic.replace(/^\//, "")}`]);
  if (normalized.endsWith("/index.html")) tokens.add(normalized.slice(0, -"/index.html".length));
  return [...tokens].filter(Boolean);
}

function referencesIn(files, candidate, matcher) {
  const tokens = candidateTokens(candidate);
  const candidateRoot = resolve(uiDir, candidate);
  return files.flatMap((file) => {
    if (file === fileURLToPath(import.meta.url)) return [];
    if (file === candidateRoot || file.startsWith(`${candidateRoot}\\`) || file.startsWith(`${candidateRoot}/`)) return [];
    const source = readText(file);
    if (!source) return [];
    const hits = tokens.filter((token) => matcher(source, token, candidate));
    return hits.length > 0 ? [`${rel(file)} -> ${hits.join(", ")}`] : [];
  });
}

function collectAudit() {
  const routeFiles = routeSourceRoots.flatMap(walk).filter((file) => /\.(?:js|mjs|cjs)$/.test(file));
  const activeBrowserFiles = browserRoots.flatMap(walk).filter((file) => /\.(?:html|js|css)$/.test(file));
  const dynamicFiles = dynamicRoots.flatMap(walk).filter((file) => /\.(?:html|js|css)$/.test(file));
  const testFiles = testRoots.flatMap(walk).filter((file) => /\.test\.(?:js|mjs|cjs)$/.test(file));

  return candidates.map((candidate) => {
    const absolute = resolve(uiDir, candidate);
    return {
      candidate,
      exists: existsSync(absolute) && (!statSync(absolute).isDirectory() || walk(absolute).length > 0),
      references: {
        expressRoutes: referencesIn(routeFiles, candidate, (source, token) => source.includes(token)),
        activeHtmlImports: referencesIn(activeBrowserFiles, candidate, (source, token) => source.includes(token)),
        dynamicJsCss: referencesIn(dynamicFiles, candidate, (source, token) => source.includes(token)),
        tests: referencesIn(testFiles, candidate, (source, token) => source.includes(token)),
        dockerfile: referencesIn(dockerFiles, candidate, (source, token) => source.includes(token)),
        readmes: referencesIn(readmeFiles, candidate, (source, token) => source.includes(token)),
      },
    };
  });
}

test("removal candidates are audited without deleting files", () => {
  const audit = collectAudit();
  assert.equal(audit.length, candidates.length);

  for (const item of audit) {
    const summary = Object.entries(item.references)
      .map(([kind, hits]) => `${kind}:${hits.length}`)
      .join(" ");
    console.log(`${item.candidate} exists:${item.exists} ${summary}`);
    for (const [kind, hits] of Object.entries(item.references)) {
      for (const hit of hits) console.log(`  ${kind} ${hit}`);
    }
  }
});
