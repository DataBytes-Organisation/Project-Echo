"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

/**
 * Small regression tests for FR-A1 duplicate prevention logic.
 *
 * These tests focus on the connection de-duplication rules used by
 * nodes-overlay.js without requiring a browser or OpenLayers runtime.
 */

function buildUniqueConnectionKeys(nodes) {
  const nodesById = new Map(
    nodes.map((node) => [String(node._id), node])
  );

  const drawnConnections = new Set();

  nodes.forEach((node) => {
    if (!Array.isArray(node.connectedNodes)) {
      return;
    }

    node.connectedNodes.forEach((connectedId) => {
      const connectedNode = nodesById.get(String(connectedId));

      if (!connectedNode) {
        return;
      }

      if (String(node._id) === String(connectedNode._id)) {
        return;
      }

      const pairKey = [
        String(node._id),
        String(connectedNode._id),
      ]
        .sort()
        .join(":");

      drawnConnections.add(pairKey);
    });
  });

  return drawnConnections;
}

test("bidirectional connections are rendered only once", () => {
  const nodes = [
    {
      _id: "A",
      connectedNodes: ["B"],
    },
    {
      _id: "B",
      connectedNodes: ["A"],
    },
  ];

  const connections = buildUniqueConnectionKeys(nodes);

  assert.equal(connections.size, 1);
  assert.ok(connections.has("A:B"));
});

test("duplicate connection ids do not create duplicate lines", () => {
  const nodes = [
    {
      _id: "A",
      connectedNodes: ["B", "B", "B"],
    },
    {
      _id: "B",
      connectedNodes: [],
    },
  ];

  const connections = buildUniqueConnectionKeys(nodes);

  assert.equal(connections.size, 1);
});

test("self-connections are ignored", () => {
  const nodes = [
    {
      _id: "A",
      connectedNodes: ["A"],
    },
  ];

  const connections = buildUniqueConnectionKeys(nodes);

  assert.equal(connections.size, 0);
});

test("missing connected nodes are ignored safely", () => {
  const nodes = [
    {
      _id: "A",
      connectedNodes: ["MISSING"],
    },
  ];

  const connections = buildUniqueConnectionKeys(nodes);

  assert.equal(connections.size, 0);
});

test("multiple logical connections remain unique", () => {
  const nodes = [
    {
      _id: "A",
      connectedNodes: ["B", "C"],
    },
    {
      _id: "B",
      connectedNodes: ["A", "C"],
    },
    {
      _id: "C",
      connectedNodes: ["A", "B"],
    },
  ];

  const connections = buildUniqueConnectionKeys(nodes);

  assert.equal(connections.size, 3);
  assert.ok(connections.has("A:B"));
  assert.ok(connections.has("A:C"));
  assert.ok(connections.has("B:C"));
});