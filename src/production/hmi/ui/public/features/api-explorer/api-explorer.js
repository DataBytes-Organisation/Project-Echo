(() => {
  const endpointEl = document.getElementById("endpoint");
  const testBtn = document.getElementById("testBtn");
  const buildBtn = document.getElementById("buildBtn");
  const clearBtn = document.getElementById("clearBtn");

  const startTs = document.getElementById("startTs");
  const endTs = document.getElementById("endTs");

  const output = document.getElementById("output");
  const statusBadge = document.getElementById("statusBadge");
  const lastChecked = document.getElementById("lastChecked");
  const respTime = document.getElementById("respTime");

  const pageState = createAdminPageState();
  pageState.resetPageState();

  function setBadge(type, text) {
    statusBadge.textContent = text;
    statusBadge.className = "badge " + type;
  }

  function buildMovementTimeEndpoint() {
    const start = (startTs.value || "").trim();
    const end = (endTs.value || "").trim();
    endpointEl.value = `/movement_time/${start}/${end}`;
  }

  buildBtn?.addEventListener("click", () => {
    buildMovementTimeEndpoint();
  });

  clearBtn?.addEventListener("click", () => {
    output.textContent = 'Click "Test" to send a request.';
    setBadge("bg-secondary", "Status: not checked");
    lastChecked.textContent = "Last checked: —";
    respTime.textContent = "Response time: —";
    pageState.hideError();
    pageState.hideLoading();
  });

  document.querySelectorAll("[data-ep]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const ep = btn.getAttribute("data-ep");
      endpointEl.value = ep;

      if (ep && ep.startsWith("/movement_time/") && startTs && endTs) {
        buildMovementTimeEndpoint();
      }
    });
  });

  testBtn?.addEventListener("click", async () => {
    const endpoint = (endpointEl.value || "").trim();

    lastChecked.textContent = "Last checked: " + new Date().toLocaleString();
    respTime.textContent = "Response time: —";
    output.textContent = `Sending GET ${endpoint}...\n`;

    const url = endpoint.startsWith("http")
      ? endpoint
      : `${window.location.origin}${endpoint.startsWith("/") ? "" : "/"}${endpoint}`;

    pageState.hideError();
    pageState.showLoading();
    setBadge("bg-warning", "Status: testing...");

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    try {
      const t0 = performance.now();
      const res = await fetch(url, {
        method: "GET",
        signal: controller.signal
      });
      const t1 = performance.now();
      const ms = Math.round(t1 - t0);

      respTime.textContent = `Response time: ${ms} ms`;

      const contentType = res.headers.get("content-type") || "";
      const raw = await res.text();

      setBadge(res.ok ? "bg-success" : "bg-danger", `Status: ${res.status}`);

      let body = raw;
      if (contentType.includes("application/json")) {
        try {
          body = JSON.stringify(JSON.parse(raw), null, 2);
        } catch (e) {
          // keep raw text if JSON parsing fails
        }
      }

      if (!res.ok) {
        pageState.showError(`Request failed with status ${res.status}.`);
      }

      output.textContent =
        `Request: GET ${url}\n` +
        `HTTP Status: ${res.status}\n` +
        `Content-Type: ${contentType || "unknown"}\n\n` +
        `Response:\n${body}`;
    } catch (err) {
      setBadge("bg-danger", "Status: failed");

      if (err.name === "AbortError") {
        pageState.showError("Request timed out. Please try again.");
      } else {
        pageState.showError("Failed to fetch API response.");
      }

      output.textContent =
        `Request: GET ${url}\n\n` +
        `Error:\n${err?.message || String(err)}`;
    } finally {
      clearTimeout(timeoutId);
      pageState.hideLoading();
    }
  });
})();