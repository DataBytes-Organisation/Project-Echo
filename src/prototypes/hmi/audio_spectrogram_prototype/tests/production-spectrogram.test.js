"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const {
  FakeAudioContext,
  FakeRoot,
  createAnimationFrameHarness,
  createDeferred,
  createResizeObserverHarness,
  createToneBuffer,
} = require("./fakes.js");

const productionModulePath = path.resolve(
  __dirname,
  "../../../../production/hmi/ui/public/js/spectrogram.js"
);
const productionWorkflowPath = path.resolve(
  __dirname,
  "../../../../production/hmi/ui/public/js/spectrogram-workflow.js"
);
const productionRecorderPath = path.resolve(
  __dirname,
  "../../../../production/hmi/ui/public/js/audio_recorder.js"
);

async function loadProductionModule() {
  const source = fs.readFileSync(productionModulePath, "utf8");
  const sourceUrl = `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
  return import(sourceUrl);
}

const subjectPromise = loadProductionModule();

async function loadWorkflowModule() {
  try {
    const source = fs.readFileSync(productionWorkflowPath, "utf8");
    const sourceUrl = `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
    return import(sourceUrl);
  } catch (error) {
    if (error && error.code === "ENOENT") return {};
    throw error;
  }
}

const workflowSubjectPromise = loadWorkflowModule();

async function loadRecorderModule() {
  const source = fs.readFileSync(productionRecorderPath, "utf8");
  const sourceUrl = `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
  return import(sourceUrl);
}

const recorderSubjectPromise = loadRecorderModule();

async function requireFunction(name) {
  const subject = await subjectPromise;
  assert.equal(typeof subject[name], "function", `${name} must be exported`);
  return subject[name];
}

async function requireWorkflowFunction(name) {
  const subject = await workflowSubjectPromise;
  assert.equal(typeof subject[name], "function", `${name} must be exported`);
  return subject[name];
}

async function createView(root, options = {}) {
  const SpectrogramView = await requireFunction("SpectrogramView");
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

test("production DSP creates deterministic visible cells with bounded frame count", async () => {
  const computeSpectrogram = await requireFunction("computeSpectrogram");
  const tone = createToneBuffer();

  const result = computeSpectrogram(tone, { fftSize: 64, hopSize: 32 });
  const repeat = computeSpectrogram(tone, { fftSize: 64, hopSize: 32 });

  assert.equal(result.duration, 0.25);
  assert.equal(result.nyquist, 4000);
  assert.equal(result.binCount, 32);
  assert.ok(result.frameCount > 1);
  assert.ok(result.frameCount <= 720);
  assert.ok(result.cells.some((value) => value > -6));
  assert.deepEqual(Array.from(result.cells), Array.from(repeat.cells));

  const longResult = computeSpectrogram(
    createToneBuffer({ duration: 10, sampleRate: 44100 }),
    { fftSize: 256, hopSize: 64 }
  );
  assert.ok(longResult.frameCount <= 720);
});

test("production PCM adapter decodes little-endian float32 samples without doubling duration", async () => {
  const decodeFloat32PcmBase64 = await requireFunction("decodeFloat32PcmBase64");
  const expected = new Float32Array([0, 0.5, -0.5, 1]);
  const encoded = Buffer.from(
    expected.buffer,
    expected.byteOffset,
    expected.byteLength
  ).toString("base64");

  const decoded = decodeFloat32PcmBase64(encoded, 4);

  assert.equal(decoded.length, 4);
  assert.equal(decoded.duration, 1);
  assert.equal(decoded.numberOfChannels, 1);
  assert.deepEqual(Array.from(decoded.getChannelData(0)), Array.from(expected));
  assert.throws(() => decoded.getChannelData(1), /channel/i);
  assert.throws(() => decodeFloat32PcmBase64("AQI=", 48000), /audio data/i);
  assert.throws(() => decodeFloat32PcmBase64(encoded, 0), /sample rate/i);
});

test("production renderer draws intensity cells with seconds, Nyquist, and dB context", async () => {
  const computeSpectrogram = await requireFunction("computeSpectrogram");
  const drawSpectrogram = await requireFunction("drawSpectrogram");
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

  assert.ok(fills.length > result.frameCount);
  assert.ok(labels.includes("0 s"));
  assert.ok(labels.includes("0.25 s"));
  assert.ok(labels.includes("0 Hz"));
  assert.ok(labels.includes("4.0 kHz"));
  assert.ok(labels.includes("Intensity (dB)"));
});

test("production decoder accepts decoded, ArrayBuffer, Blob, and File-shaped inputs", async (t) => {
  const AudioDecoder = await requireFunction("AudioDecoder");
  const decoded = createToneBuffer();
  const directDecoder = new AudioDecoder({
    createAudioContext() {
      throw new Error("AudioContext must not be opened for decoded data");
    },
  });
  assert.equal(await directDecoder.decode(decoded), decoded);

  const bytes = new Uint8Array([82, 73, 70, 70]).buffer;
  const inputs = [
    ["ArrayBuffer", bytes],
    ["Blob", new Blob([bytes], { type: "audio/wav" })],
    ["File-shaped", {
      name: "clip.wav",
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
          const context = new FakeAudioContext({ decodeResult: decoded });
          contexts.push(context);
          return context;
        },
      });

      assert.equal(await decoder.decode(input), decoded);
      assert.equal(contexts.length, 1);
      assert.equal(contexts[0].closeCalls, 1);
      await decoder.destroy();
    });
  }
});

test("production view replaces loading with success or a sanitized decode error", async () => {
  const successRoot = new FakeRoot();
  const deferred = createDeferred();
  const successHarness = await createView(successRoot, {
    decoder: {
      decode: () => deferred.promise,
      destroy: async () => {},
    },
  });

  assert.equal(successRoot.dataset.state, "empty");
  const pending = successHarness.view.load(new ArrayBuffer(8));
  assert.equal(successRoot.dataset.state, "loading");
  deferred.resolve(createToneBuffer());
  await pending;
  successHarness.animation.flushAll();
  assert.equal(successRoot.dataset.state, "success");

  const errorRoot = new FakeRoot();
  const errorHarness = await createView(errorRoot, {
    decoder: {
      decode: async () => {
        throw new Error("EncodingError at http://internal.example/audio\nraw stack");
      },
      destroy: async () => {},
    },
  });
  assert.equal(await errorHarness.view.load(new ArrayBuffer(8)), null);
  assert.equal(errorRoot.dataset.state, "error");
  const message = errorRoot.elements['[data-role="error"]'].textContent;
  assert.match(message, /couldn't decode this audio clip/i);
  assert.doesNotMatch(message, /EncodingError|https?:|stack|internal/i);

  await successHarness.view.destroy();
  await errorHarness.view.destroy();
});

for (const panel of [
  { height: 300, name: "animal", resizedHeight: 260, resizedWidth: 540, width: 720 },
  { height: 240, name: "microphone", resizedHeight: 300, resizedWidth: 620, width: 420 },
]) {
  test(`production ${panel.name} view tracks visible size in its canvas backing store`, async () => {
    const root = new FakeRoot(panel.width, panel.height);
    const harness = await createView(root, { devicePixelRatio: 2 });

    await harness.view.load(createToneBuffer());
    harness.animation.flushAll();
    assert.equal(root.canvas.width, panel.width * 2);
    assert.equal(root.canvas.height, panel.height * 2);

    root.setSize(panel.resizedWidth, panel.resizedHeight);
    harness.resize.instances[0].trigger();
    harness.animation.flushAll();
    assert.equal(root.canvas.width, panel.resizedWidth * 2);
    assert.equal(root.canvas.height, panel.resizedHeight * 2);
    await harness.view.destroy();
  });
}

test("production view ignores stale loads and destroys contexts, observers, and frames", async () => {
  const AudioDecoder = await requireFunction("AudioDecoder");
  const slow = createDeferred();
  const decoder = {
    decode(input) {
      return input === "slow" ? slow.promise : Promise.resolve(createToneBuffer());
    },
    destroyCalls: 0,
    async destroy() {
      this.destroyCalls += 1;
    },
  };
  const root = new FakeRoot();
  const harness = await createView(root, { decoder });

  const staleLoad = harness.view.load("slow");
  await harness.view.load("new");
  slow.reject(new Error("stale internal failure"));
  await staleLoad;
  assert.equal(root.dataset.state, "success");

  harness.resize.instances[0].trigger();
  assert.equal(harness.animation.pendingCount, 1);
  await harness.view.destroy();

  assert.equal(decoder.destroyCalls, 1);
  assert.equal(harness.resize.instances[0].disconnected, true);
  assert.equal(harness.animation.pendingCount, 0);
  assert.equal(harness.animation.cancelled.length, 1);

  const decodeDeferred = createDeferred();
  const contexts = [];
  const contextDecoder = new AudioDecoder({
    createAudioContext() {
      const context = new FakeAudioContext({ decodeDeferred });
      contexts.push(context);
      return context;
    },
  });
  void contextDecoder.decode(new ArrayBuffer(8));
  await Promise.resolve();
  await contextDecoder.destroy();
  assert.equal(contexts[0].closeCalls, 1);
});

test("animal workflow retrieves once, shares decoded audio, and ignores stale selections", async () => {
  const AnimalSpectrogramWorkflow = await requireWorkflowFunction("AnimalSpectrogramWorkflow");
  const requests = new Map();
  const routeCalls = [];
  const rendered = [];
  const states = [];
  const view = {
    destroyCalls: 0,
    async destroy() { this.destroyCalls += 1; },
    async load(decoded) { rendered.push(decoded); return { frameCount: 1 }; },
    showDecodeError() { states.push("decode-error"); },
    showEmpty() { states.push("empty"); },
    showLoading() { states.push("loading"); },
    showLoadError() { states.push("load-error"); },
  };
  const workflow = new AnimalSpectrogramWorkflow({
    decodePcm(encoded, sampleRate) {
      return { encoded, sampleRate };
    },
    retrieveAudio(eventId) {
      routeCalls.push(eventId);
      const request = createDeferred();
      requests.set(eventId, request);
      return request.promise;
    },
    view,
  });

  const slowLoad = workflow.select("slow");
  const currentLoad = workflow.select("current");
  requests.get("current").resolve({ data: { audioClip: "CURRENT", sampleRate: 48000 } });
  await currentLoad;
  requests.get("slow").resolve({ data: { audioClip: "SLOW", sampleRate: 16000 } });
  await slowLoad;

  assert.deepEqual(routeCalls, ["slow", "current"]);
  assert.deepEqual(rendered, [{ encoded: "CURRENT", sampleRate: 48000 }]);
  assert.deepEqual(await workflow.getSelectedAudio(), { encoded: "CURRENT", sampleRate: 48000 });
  assert.equal(workflow.selectedEventId, "current");
  assert.deepEqual(states, ["loading", "loading"]);

  workflow.clear();
  assert.equal(await workflow.getSelectedAudio(), null);
  assert.equal(states.at(-1), "empty");
  await workflow.destroy();
  assert.equal(view.destroyCalls, 1);
});

test("animal workflow distinguishes route and PCM decode errors without exposing raw details", async () => {
  const AnimalSpectrogramWorkflow = await requireWorkflowFunction("AnimalSpectrogramWorkflow");
  const states = [];
  const view = {
    async destroy() {},
    async load() {},
    showDecodeError() { states.push("decode-error"); },
    showEmpty() {},
    showLoading() {},
    showLoadError() { states.push("load-error"); },
  };
  const routeFailure = new AnimalSpectrogramWorkflow({
    decodePcm: () => null,
    retrieveAudio: async () => { throw new Error("http://internal/api raw response"); },
    view,
  });
  await routeFailure.select("route-error");

  const decodeFailure = new AnimalSpectrogramWorkflow({
    decodePcm: () => { throw new Error("private decoder stack"); },
    retrieveAudio: async () => ({ data: { audioClip: "bad", sampleRate: 48000 } }),
    view,
  });
  await decodeFailure.select("decode-error");

  assert.deepEqual(states, ["load-error", "decode-error"]);
});

test("microphone workflow decodes once, shares the result, clears, and destroys resources", async () => {
  const MicrophoneSpectrogramWorkflow = await requireWorkflowFunction("MicrophoneSpectrogramWorkflow");
  const decoded = createToneBuffer();
  const input = new Blob([new Uint8Array([1, 2, 3, 4])], { type: "audio/webm" });
  const decodeInputs = [];
  const rendered = [];
  const states = [];
  const decoder = {
    destroyCalls: 0,
    async decode(value) { decodeInputs.push(value); return decoded; },
    async destroy() { this.destroyCalls += 1; },
  };
  const view = {
    destroyCalls: 0,
    async destroy() { this.destroyCalls += 1; },
    async load(value) { rendered.push(value); return { frameCount: 1 }; },
    showDecodeError() { states.push("decode-error"); },
    showEmpty() { states.push("empty"); },
    showLoading() { states.push("loading"); },
  };
  const workflow = new MicrophoneSpectrogramWorkflow({ decoder, view });

  assert.equal(await workflow.load(input), decoded);
  assert.deepEqual(decodeInputs, [input]);
  assert.deepEqual(rendered, [decoded]);
  assert.equal(await workflow.getSelectedAudio(), decoded);
  assert.deepEqual(states, ["loading"]);

  workflow.clear();
  assert.equal(await workflow.getSelectedAudio(), null);
  assert.equal(states.at(-1), "empty");
  await workflow.destroy();
  assert.equal(decoder.destroyCalls, 1);
  assert.equal(view.destroyCalls, 1);
});

test("latest microphone source guard rejects every superseded operation", async () => {
  const LatestSourceGuard = await requireWorkflowFunction("LatestSourceGuard");
  const guard = new LatestSourceGuard();

  const fileSelection = guard.begin();
  assert.equal(guard.isCurrent(fileSelection), true);

  const recordingStop = guard.begin();
  assert.equal(guard.isCurrent(fileSelection), false);
  assert.equal(guard.isCurrent(recordingStop), true);

  guard.invalidate();
  assert.equal(guard.isCurrent(recordingStop), false);
});

test("recorder setup failure stops a microphone stream that was already granted", async () => {
  const { getAudioRecorder } = await recorderSubjectPromise;
  const tracks = [{ stopCalls: 0, stop() { this.stopCalls += 1; } }];
  const stream = { getTracks: () => tracks };
  class ThrowingMediaRecorder {
    constructor() {
      throw new Error("MediaRecorder setup failed");
    }
  }
  const recorder = getAudioRecorder({
    MediaRecorderClass: ThrowingMediaRecorder,
    getUserMedia: async () => stream,
  });

  await assert.rejects(() => recorder.start(), /MediaRecorder setup failed/);
  assert.equal(tracks[0].stopCalls, 1);
  assert.equal(recorder.mediaRecorder, null);
  assert.equal(recorder.streamBeingCaptured, null);
});
