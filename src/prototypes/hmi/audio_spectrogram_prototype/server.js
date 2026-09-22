"use strict";

const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");

const host = "127.0.0.1";
const port = Number.parseInt(process.env.PORT || "4173", 10);
const publicFiles = new Map([
  ["/", ["index.html", "text/html; charset=utf-8"]],
  ["/index.html", ["index.html", "text/html; charset=utf-8"]],
  ["/styles.css", ["styles.css", "text/css; charset=utf-8"]],
  ["/spectrogram.js", ["spectrogram.js", "text/javascript; charset=utf-8"]],
  ["/demo.js", ["demo.js", "text/javascript; charset=utf-8"]],
]);

const server = http.createServer((request, response) => {
  const pathname = new URL(request.url, `http://${host}:${port}`).pathname;
  const publicFile = publicFiles.get(pathname);

  if (!publicFile) {
    response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    response.end("Not found");
    return;
  }

  const [filename, contentType] = publicFile;
  fs.readFile(path.join(__dirname, filename), (error, content) => {
    if (error) {
      response.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
      response.end("Prototype file unavailable");
      return;
    }

    response.writeHead(200, {
      "Cache-Control": "no-store",
      "Content-Type": contentType,
    });
    response.end(content);
  });
});

server.listen(port, host, () => {
  process.stdout.write(`FR-B1 spectrogram prototype: http://${host}:${port}\n`);
});
