"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const {
  FakeAudioContext,
  FakeRoot,
  createAnimationFrameHarness,
  createDeferred,
  createResizeObserverHarness,
  createToneBuffer,
} = require("./fakes.js");

function loadSubject() {
  try {
    return require("../spectrogram.js");
  } catch (error) {
    if (error && error.code === "MODULE_NOT_FOUND") {
      return {};
    }
    throw error;
  }
}

const subject = loadSubject();

function loadDemoSubject() {
  try {
    return require("../demo.js");
  } catch (error) {
    if (error instanceof ReferenceError && /window is not defined/.test(error.message)) {
      return {};
    }
    throw error;
  }
}

const demoSubject = loadDemoSubject();

function requireFunction(name) {
  assert.equal(typeof subject[name], "function", `${name} must be exported`);
  return subject[name];
}

function requireDemoFunction(name) {
  assert.equal(typeof demoSubject[name], "function", `${name} must be exported`);
  return demoSubject[name];
}

function createView(root, options = {}) {
  const SpectrogramView = requireFunction("SpectrogramView");
  const resize = options.resize || createResizeObserverHarness();
  const animation = options.animation || createAnimationFrameHarness();
  const decoder = options.decoder || {
    decode: async (input) => input,
    destroy: async () => {},
  };

  const view = new SpectrogramView(root, {
    ResizeObserverClass: resize.ResizeObserverClass,
    cancelAnimationFrame: animation.cancelAnimationFrame,
    decoder,
    devicePixelRatio: options.devicePixelRatio || 1,
    fftOptions: { fftSize: 64, hopSize: 32 },
    requestAnimationFrame: animation.requestAnimationFrame,
  });

  return { animation, decoder, resize, view };
}

test("valid decoded audio produces deterministic non-empty frequency cells", () => {
  const computeSpectrogram = requireFunction("computeSpectrogram");
  const audioBuffer = createToneBuffer();

  const result = computeSpectrogram(audioBuffer, { fftSize: 64, hopSize: 32 });

  assert.equal(result.sampleRate, 8000);
  assert.equal(result.duration, 0.25);
  assert.equal(result.nyquist, 4000);
  assert.equal(result.binCount, 32);
  assert.ok(result.frameCount > 1);
  assert.equal(result.cells.length, result.frameCount * result.binCount);
  assert.ok(result.cells.some((value) => value > -6), "tone must create visible energy");

  const averageByBin = new Float64Array(result.binCount);
  for (let frame = 0; frame < result.frameCount; frame += 1) {
    for (let bin = 0; bin < result.binCount; bin += 1) {
      averageByBin[bin] += result.cells[frame * result.binCount + bin];
    }
  }
  const peakBin = averageByBin.indexOf(Math.max(...averageByBin));
  assert.equal(peakBin, 8, "1 kHz must peak in FFT bin 8 at 8 kHz / 64 samples");
});

test("long clips cap time frames to keep computation and redraw work bounded", () => {
  const computeSpectrogram = requireFunction("computeSpectrogram");
  const audioBuffer = createToneBuffer({ duration: 10, sampleRate: 44100 });

  const result = computeSpectrogram(audioBuffer, { fftSize: 256, hopSize: 64 });

  assert.ok(result.frameCount <= 720);
  assert.ok(result.cells.length <= 720 * 128);
  assert.equal(result.duration, 10);
  assert.equal(result.nyquist, 22050);
});

test("canvas renderer draws intensity cells and readable time, Nyquist, and dB context", () => {
  const computeSpectrogram = requireFunction("computeSpectrogram");
  const drawSpectrogram = requireFunction("drawSpectrogram");
  const root = new FakeRoot(640, 280);
  const result = computeSpectrogram(createToneBuffer(), { fftSize: 64, hopSize: 32 });

  drawSpectrogram(root.canvas, result, {
    devicePixelRatio: 1,
    displayHeight: 280,
    displayWidth: 640,
  });

  const fills = root.canvas.context.operations.filter(({ name }) => name === "fillRect");
  const labels = root.canvas.context.operations
    .filter(({ name }) => name === "fillText")
    .map(({ args }) => args[0]);

  assert.equal(root.canvas.width, 640);
  assert.equal(root.canvas.height, 280);
  assert.ok(fills.length > result.frameCount, "intensity cells must be painted");
  assert.ok(labels.includes("0 s"));
  assert.ok(labels.includes("0.25 s"));
  assert.ok(labels.includes("0 Hz"));
  assert.ok(labels.includes("4.0 kHz"));
  assert.ok(labels.includes("Intensity (dB)"));
});

test("decoder returns AudioBuffer-like input without opening an audio context", async () => {
  const AudioDecoder = requireFunction("AudioDecoder");
  const decoded = createToneBuffer();
  const decoder = new AudioDecoder({
    createAudioContext() {
      throw new Error("AudioContext must not be created for decoded input");
    },
  });

  assert.equal(await decoder.decode(decoded), decoded);
  await decoder.destroy();
});

test("decoder accepts ArrayBuffer, Blob, and File-shaped inputs and closes each context", async (t) => {
  const AudioDecoder = requireFunction("AudioDecoder");
  const expected = createToneBuffer();
  const bytes = new Uint8Array([82, 73, 70, 70]).buffer;
  const inputs = [
    ["ArrayBuffer", bytes],
    ["Blob", new Blob([bytes], { type: "audio/wav" })],
    ["File-shaped", {
      name: "tone.wav",
      size: bytes.byteLength,
      type: "audio/wav",
      arrayBuffer: async () => bytes.slice(0),
    }],
  ];

  for (const [name, input] of inputs) {
    await t.test(name, async () => {
      const contexts = [];
      const decoder = new AudioDecoder({
        createAudioContext() {
          const context = new FakeAudioContext({ decodeResult: expected });
          contexts.push(context);
          return context;
        },
      });

      assert.equal(await decoder.decode(input), expected);
      assert.equal(contexts.length, 1);
      assert.ok(contexts[0].decodeInputs[0] instanceof ArrayBuffer);
      assert.equal(contexts[0].closeCalls, 1);
      await decoder.destroy();
    });
  }
});

test("component exposes empty, loading, and drawn success states", async () => {
  const root = new FakeRoot(620, 270);
  const deferred = createDeferred();
  const decoder = {
    decode: () => deferred.promise,
    destroy: async () => {},
  };
  const { animation, view } = createView(root, { decoder });

  assert.equal(root.dataset.state, "empty");
  assert.equal(root.elements['[data-role="empty"]'].hidden, false);

  const loadPromise = view.load(new ArrayBuffer(16));
  assert.equal(root.dataset.state, "loading");
  assert.equal(root.elements['[data-role="loading"]'].hidden, false);

  deferred.resolve(createToneBuffer());
  const result = await loadPromise;
  animation.flushAll();

  assert.equal(root.dataset.state, "success");
  assert.equal(root.elements['[data-role="loading"]'].hidden, true);
  assert.equal(result.nyquist, 4000);
  assert.ok(root.canvas.context.operations.some(({ name }) => name === "fillRect"));
  await view.destroy();
});

test("decode failure replaces loading with a sanitized user-facing error", async () => {
  const root = new FakeRoot();
  const decoder = {
    decode: async () => {
      throw new Error("EncodingError: decoder at http://internal.example/audio\nraw stack");
    },
    destroy: async () => {},
  };
  const { view } = createView(root, { decoder });

  const result = await view.load(new ArrayBuffer(8));

  assert.equal(result, null);
  assert.equal(root.dataset.state, "error");
  assert.equal(root.elements['[data-role="loading"]'].hidden, true);
  assert.equal(root.elements['[data-role="error"]'].hidden, false);
  assert.match(root.elements['[data-role="error"]'].textContent, /couldn't decode this audio clip/i);
  assert.doesNotMatch(root.elements['[data-role="error"]'].textContent, /EncodingError|https?:|stack|internal/i);
  await view.destroy();
});

test("clearing a rendered clip removes its stale canvas description", async () => {
  const root = new FakeRoot();
  const harness = createView(root);

  await harness.view.load(createToneBuffer());
  harness.animation.flushAll();
  assert.match(root.canvas.getAttribute("aria-label"), /Spectrogram from 0 to/);

  harness.view.showEmpty();
  harness.animation.flushAll();

  assert.equal(root.canvas.getAttribute("aria-label"), null);
  await harness.view.destroy();
});

for (const panel of [
  { height: 300, name: "animal", resizedHeight: 260, resizedWidth: 540, width: 720 },
  { height: 240, name: "microphone", resizedHeight: 300, resizedWidth: 620, width: 420 },
]) {
  test(`${panel.name} panel tracks visible dimensions in its canvas backing store`, async () => {
    const root = new FakeRoot(panel.width, panel.height);
    const harness = createView(root, { devicePixelRatio: 2 });

    await harness.view.load(createToneBuffer());
    harness.animation.flushAll();
    assert.equal(root.canvas.width, panel.width * 2);
    assert.equal(root.canvas.height, panel.height * 2);
    assert.equal(root.canvas.style.width, `${panel.width}px`);
    assert.equal(root.canvas.style.height, `${panel.height}px`);

    root.setSize(panel.resizedWidth, panel.resizedHeight);
    harness.resize.instances[0].trigger();
    harness.animation.flushAll();
    assert.equal(root.canvas.width, panel.resizedWidth * 2);
    assert.equal(root.canvas.height, panel.resizedHeight * 2);
    await harness.view.destroy();
  });
}

test("destroy closes pending audio contexts, disconnects resize observation, and cancels frames", async () => {
  const AudioDecoder = requireFunction("AudioDecoder");
  const decodeDeferred = createDeferred();
  const contexts = [];
  const decoder = new AudioDecoder({
    createAudioContext() {
      const context = new FakeAudioContext({ decodeDeferred });
      contexts.push(context);
      return context;
    },
  });
  const root = new FakeRoot();
  const harness = createView(root, { decoder });

  void harness.view.load(new ArrayBuffer(16));
  await Promise.resolve();
  await Promise.resolve();
  harness.resize.instances[0].trigger();

  assert.equal(contexts.length, 1);
  assert.equal(harness.animation.pendingCount, 1);

  await harness.view.destroy();

  assert.equal(contexts[0].closeCalls, 1);
  assert.equal(harness.resize.instances[0].disconnected, true);
  assert.equal(harness.animation.pendingCount, 0);
  assert.equal(harness.animation.cancelled.length, 1);
});

test("audio contexts close when browser decoding fails", async () => {
  const AudioDecoder = requireFunction("AudioDecoder");
  const contexts = [];
  const decoder = new AudioDecoder({
    createAudioContext() {
      const context = new FakeAudioContext({ decodeError: new Error("EncodingError") });
      contexts.push(context);
      return context;
    },
  });

  await assert.rejects(() => decoder.decode(new ArrayBuffer(8)), /EncodingError/);
  assert.equal(contexts[0].closeCalls, 1);
  await decoder.destroy();
});

test("incompatible created audio contexts are closed before decoder rejection", async () => {
  const AudioDecoder = requireFunction("AudioDecoder");
  const context = {
    closeCalls: 0,
    close() {
      this.closeCalls += 1;
      return Promise.resolve();
    },
  };
  const decoder = new AudioDecoder({ createAudioContext: () => context });

  await assert.rejects(() => decoder.decode(new ArrayBuffer(8)), /compatible AudioContext/);
  assert.equal(context.closeCalls, 1);
  await decoder.destroy();
  assert.equal(context.closeCalls, 1);
});

test("a stale panel load cannot overwrite the status from a newer load", async () => {
  const createPanelLoader = requireDemoFunction("createPanelLoader");
  const slow = createDeferred();
  const messages = [];
  const views = [{
    load(input) {
      return input === "slow" ? slow.promise : Promise.resolve({ frameCount: 1 });
    },
  }];
  const loader = createPanelLoader(views, (message) => messages.push(message));

  const staleLoad = loader.load("slow", "Decoding slow clip", "Slow clip ready");
  await loader.load("new", "Decoding new clip", "New clip ready");
  slow.resolve(null);
  await staleLoad;

  assert.deepEqual(messages, [
    "Decoding slow clip",
    "Decoding new clip",
    "New clip ready",
  ]);
});
