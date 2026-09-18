import assert from "node:assert/strict";
import { existsSync, readFileSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const uiDir = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const publicDir = join(uiDir, "public");
const sidebarFile = join(publicDir, "shared/admin/components/sidebar-component.html");

// Read the pairs from the manifest source instead of importing its test file,
// so running this file does not re-register the manifest test cases.
const manifestSource = readFileSync(join(uiDir, "tests/server/page-route-manifest.test.mjs"), "utf8");
const pageRoutes = [...manifestSource.matchAll(/\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*(?:true|false)\s*\]/g)].map(
  ([, route, file]) => [route, file]
);
const routeFiles = new Map(pageRoutes);
assert.ok(pageRoutes.length > 0, "should parse at least one page route from the manifest");

const localAttribute = /(?:src|href)=["']([^"'#?]+)["']/g;
const moduleImport = /(?:import|export)\s+(?:[\s\S]*?\s+from\s+)?["']([^"']+)["']/g;
const cssUrl = /url\(["']?([^"')?#]+)["']?\)/g;
const cssImport = /@import\s+["']([^"'#?]+)["']/g;

// Already-broken references. Fails on any new missing local dependency and on
// any entry here that starts resolving again.
const expectedMissing = new Set([
  "public/assets/styles/HMI.css", // referenced by public/pages/map/index.html
  "public/assets/styles/HMI-menu.css", // referenced by public/pages/map/index.html
  "public/assets/styles/HMI-controls.css", // referenced by public/pages/map/index.html
  "public/assets/styles/HMI-audio.css", // referenced by public/pages/map/index.html
  "public/assets/styles/HMI-weather.css", // referenced by public/pages/map/index.html
  "public/js/audio.js", // referenced by public/pages/auth/login.html and reset-password.html
  "public/vendor/libs/simplebar/dist/simplebar.css", // pre-existing unresolved @import in public/vendor/admin/styles.min.css
]);

function isIgnored(raw) {
  const value = raw.trim();
  if (value === "") return true;
  if (/^(https?:|data:|mailto:|tel:|blob:)/i.test(value)) return true;
  if (value.startsWith("#")) return true;
  if (value.includes("${") || value.includes("{{") || value.includes("<%")) return true;
  const clean = value.split("#")[0].split("?")[0];
  const base = clean.split("/").pop();
  if (!base || !base.includes(".")) return true; // extensionless route, not a file
  return false;
}

function cleanRef(raw) {
  let clean = raw.trim().split("#")[0].split("?")[0].trim();
  try {
    if (clean.includes("%")) clean = decodeURIComponent(clean);
  } catch {
    // keep the raw spelling when it is not valid percent-encoding
  }
  return clean;
}

function resolveFromPageUrl(ref, pageUrl) {
  if (ref.startsWith("/")) return join(publicDir, ref.slice(1));
  const dir = pageUrl.endsWith("/") ? pageUrl : `${pageUrl.slice(0, pageUrl.lastIndexOf("/") + 1)}`;
  const pathname = new URL(ref, `http://local${dir.startsWith("/") ? dir : `/${dir}`}`).pathname;
  return join(publicDir, decodeURIComponent(pathname).slice(1));
}

function toPublicPath(absolute) {
  return absolute.split(publicDir)[1] ?? null;
}

function stripJsComments(source) {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/^\s*\/\/.*$/gm, "");
}

function collectActiveGraph() {
  const entries = new Map();
  for (const [route, file] of pageRoutes) {
    if (!entries.has(route)) entries.set(route, { pageUrl: route, file });
  }
  const sidebar = readFileSync(sidebarFile, "utf8");
  for (const match of sidebar.matchAll(/href=["']([^"'#?]+)["']/g)) {
    if (!match[1].endsWith(".html")) continue;
    const file = `public${match[1]}`;
    if (!entries.has(match[1])) entries.set(match[1], { pageUrl: match[1], file });
  }

  const missing = new Map();
  const visited = new Set();

  function note(absolute, referrer) {
    const key = absolute.startsWith(uiDir)
      ? absolute.slice(uiDir.length + 1).replace(/\\/g, "/")
      : absolute;
    if (!missing.has(key)) missing.set(key, []);
    missing.get(key).push(referrer);
  }

  function crawlFile(absolute, referrer, pageUrl, isEntry) {
    if (!existsSync(absolute) || !statSync(absolute).isFile()) {
      note(absolute, referrer);
      return;
    }
    if (absolute.endsWith(".html") && !isEntry) return; // existence-checked only
    if (visited.has(absolute)) return;
    visited.add(absolute);
    let content = readFileSync(absolute, "utf8");
    const relative = absolute.slice(uiDir.length + 1).replace(/\\/g, "/");
    const isCss = absolute.endsWith(".css");
    const isJs = absolute.endsWith(".js");
    if (!isCss && !isJs) content = content.replace(/<!--[\s\S]*?-->/g, "");
    const patterns = isCss ? [cssUrl, cssImport] : isJs ? [moduleImport] : [localAttribute, moduleImport];
    for (const pattern of patterns) {
      pattern.lastIndex = 0;
      let found;
      while ((found = pattern.exec(content)) !== null) {
        if (isIgnored(found[1])) continue;
        const ref = cleanRef(found[1]);
        if (ref === "") continue;
        let target;
        if (routeFiles.has(ref)) {
          target = join(uiDir, routeFiles.get(ref));
        } else if (ref.startsWith("/")) {
          target = join(publicDir, ref.slice(1));
        } else if (isCss || isJs) {
          target = resolve(dirname(absolute), ref);
        } else {
          target = resolveFromPageUrl(ref, pageUrl);
        }
        if (toPublicPath(target) === null || toPublicPath(target).startsWith("..")) continue;
        crawlFile(target, `${relative} -> ${found[1].trim()}`, pageUrl, false);
      }
    }
  }

  for (const { pageUrl, file } of entries.values()) {
    crawlFile(join(uiDir, file), `entry ${pageUrl}`, pageUrl, true);
  }
  return { missing, visited };
}

test("active pages have no new missing local dependencies", () => {
  const { missing } = collectActiveGraph();
  const unexpected = [...missing.keys()].filter((key) => !expectedMissing.has(key));
  assert.deepEqual(
    unexpected,
    [],
    `new missing local dependencies:\n${unexpected
      .map((key) => `  ${key} (referenced by ${missing.get(key).join("; ")})`)
      .join("\n")}`
  );
});

test("documented missing dependencies are still missing", () => {
  const stale = [...expectedMissing].filter((key) => existsSync(join(uiDir, key)));
  assert.deepEqual(stale, [], `these files exist now, remove them from expectedMissing: ${stale.join(", ")}`);
});

test("active browser files do not call direct FastAPI origins", () => {
  const { visited } = collectActiveGraph();
  const offenders = [...visited]
    .filter((file) => /\.(?:html|js|css)$/.test(file))
    .flatMap((file) => {
      const relative = file.slice(uiDir.length + 1).replace(/\\/g, "/");
      const content = stripJsComments(readFileSync(file, "utf8"));
      return /https?:\/\/[^\s"'`]+:9000|http:\/\/localhost:9000/.test(content) ? [relative] : [];
    });

  assert.deepEqual(offenders, []);
});
