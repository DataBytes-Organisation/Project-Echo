"use strict";

/**
 * detection_stream_client.js
 * Connects the HMI map to the Backend detection WebSocket stream.
 *
 * Deploy note: set window.ECHO_API_WS_URL (e.g. wss://api.example.com) in
 * non-local environments. The localhost fallback is for local Docker only.
 */

function resolveWsBase() {
  if (window.ECHO_API_WS_URL) {
    return window.ECHO_API_WS_URL;
  }

  // Local Docker default: HMI on :8080, API on :9000
  if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
    return "ws://localhost:9000";
  }

  // Same-host deploy: derive secure WS from the current page origin
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}`;
}

function resolveAuthToken() {
  try {
    return window.localStorage.getItem("token");
  } catch (error) {
    console.error("[B1.2 WS] unable to read auth token", error);
    return null;
  }
}

export function connectDetectionStream({ onDetection, onStatus }) {
  const token = resolveAuthToken();
  if (!token) {
    console.error("[B1.2 WS] missing login token; detection stream not started");
    onStatus?.("unauthorized");
    return () => {};
  }

  const wsBase = resolveWsBase();
  const wsUrl = `${wsBase}/ws/detections?token=${encodeURIComponent(token)}`;
  const ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log("[B1.2 WS] connected to", `${wsBase}/ws/detections`);
    onStatus?.("connected");
  };
  ws.onclose = (event) => {
    console.log("[B1.2 WS] disconnected", event.code);
    onStatus?.(event.code === 1008 ? "unauthorized" : "closed");
  };
  ws.onerror = () => {
    console.error("[B1.2 WS] connection error");
    onStatus?.("error");
  };
  ws.onmessage = (event) => {
    try {
      const detection = JSON.parse(event.data);
      console.log("[B1.2 WS] STREAM RECEIVED:", detection);
      onDetection(detection);
    } catch (error) {
      console.error("[B1.2 WS] invalid detection stream payload", error);
    }
  };

  return () => {
    if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
      ws.close();
    }
  };
}
