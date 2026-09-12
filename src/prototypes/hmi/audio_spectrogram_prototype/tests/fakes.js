"use strict";

function createDeferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function createToneBuffer({
  frequency = 1000,
  sampleRate = 8000,
  duration = 0.25,
  amplitude = 0.8,
} = {}) {
  const length = Math.round(sampleRate * duration);
  const samples = new Float32Array(length);

  for (let index = 0; index < length; index += 1) {
    samples[index] = amplitude * Math.sin(2 * Math.PI * frequency * index / sampleRate);
  }

  return {
    duration: length / sampleRate,
    length,
    numberOfChannels: 1,
    sampleRate,
    getChannelData(channel) {
      if (channel !== 0) {
        throw new RangeError("Only channel zero exists in this fixture.");
      }
      return samples;
    },
  };
}

class FakeCanvasContext {
  constructor() {
    this.fillStyle = "#000000";
    this.font = "12px sans-serif";
    this.globalAlpha = 1;
    this.lineWidth = 1;
    this.strokeStyle = "#000000";
    this.textAlign = "start";
    this.textBaseline = "alphabetic";
    this.operations = [];
  }

  record(name, args) {
    this.operations.push({
      args: Array.from(args),
      fillStyle: this.fillStyle,
      font: this.font,
      name,
      textAlign: this.textAlign,
      textBaseline: this.textBaseline,
    });
  }

  clearRect(...args) { this.record("clearRect", args); }
  fillRect(...args) { this.record("fillRect", args); }
  fillText(...args) { this.record("fillText", args); }
  restore(...args) { this.record("restore", args); }
  save(...args) { this.record("save", args); }
  setTransform(...args) { this.record("setTransform", args); }
  strokeRect(...args) { this.record("strokeRect", args); }

  measureText(text) {
    return { width: String(text).length * 7 };
  }
}

class FakeElement {
  constructor() {
    this.attributes = new Map();
    this.dataset = {};
    this.hidden = false;
    this.style = {};
    this.textContent = "";
  }

  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }

  removeAttribute(name) {
    this.attributes.delete(name);
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }
}

class FakeCanvas extends FakeElement {
  constructor() {
    super();
    this.context = new FakeCanvasContext();
    this.height = 0;
    this.width = 0;
  }

  getContext(kind) {
    return kind === "2d" ? this.context : null;
  }
}

class FakeRoot extends FakeElement {
  constructor(width = 640, height = 280) {
    super();
    this.canvas = new FakeCanvas();
    this.viewport = new FakeElement();
    this.viewport.clientHeight = height;
    this.viewport.clientWidth = width;
    this.elements = {
      '[data-role="canvas"]': this.canvas,
      '[data-role="empty"]': new FakeElement(),
      '[data-role="error"]': new FakeElement(),
      '[data-role="loading"]': new FakeElement(),
      '[data-role="summary"]': new FakeElement(),
      '[data-role="viewport"]': this.viewport,
    };
  }

  querySelector(selector) {
    return this.elements[selector] || null;
  }

  setSize(width, height) {
    this.viewport.clientWidth = width;
    this.viewport.clientHeight = height;
  }
}

function createAnimationFrameHarness() {
  const callbacks = new Map();
  const cancelled = [];
  let nextId = 1;

  return {
    cancelAnimationFrame(id) {
      cancelled.push(id);
      callbacks.delete(id);
    },
    cancelled,
    flushAll() {
      const queued = Array.from(callbacks.entries());
      callbacks.clear();
      for (const [, callback] of queued) {
        callback(16);
      }
    },
    get pendingCount() {
      return callbacks.size;
    },
    requestAnimationFrame(callback) {
      const id = nextId;
      nextId += 1;
      callbacks.set(id, callback);
      return id;
    },
  };
}

function createResizeObserverHarness() {
  const instances = [];

  class FakeResizeObserver {
    constructor(callback) {
      this.callback = callback;
      this.disconnected = false;
      this.observed = [];
      instances.push(this);
    }

    disconnect() {
      this.disconnected = true;
      this.observed = [];
    }

    observe(element) {
      this.observed.push(element);
    }

    trigger() {
      this.callback(this.observed.map((target) => ({ target })));
    }
  }

  return { instances, ResizeObserverClass: FakeResizeObserver };
}

class FakeAudioContext {
  constructor({ decodeDeferred, decodeError, decodeResult } = {}) {
    this.closeCalls = 0;
    this.decodeDeferred = decodeDeferred;
    this.decodeError = decodeError;
    this.decodeInputs = [];
    this.decodeResult = decodeResult;
  }

  close() {
    this.closeCalls += 1;
    return Promise.resolve();
  }

  decodeAudioData(input) {
    this.decodeInputs.push(input);
    if (this.decodeDeferred) {
      return this.decodeDeferred.promise;
    }
    if (this.decodeError) {
      return Promise.reject(this.decodeError);
    }
    return Promise.resolve(this.decodeResult);
  }
}

module.exports = {
  FakeAudioContext,
  FakeRoot,
  createAnimationFrameHarness,
  createDeferred,
  createResizeObserverHarness,
  createToneBuffer,
};
