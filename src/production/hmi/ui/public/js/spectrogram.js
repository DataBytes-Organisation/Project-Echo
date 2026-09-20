"use strict";

export const AUDIO_DECODE_ERROR_MESSAGE = "We couldn't decode this audio clip. Choose another recording and try again.";
export const AUDIO_LOAD_ERROR_MESSAGE = "This audio clip is currently unavailable.";

const DEFAULT_FFT_SIZE = 1024;
const DEFAULT_HOP_SIZE = 256;
const MAX_TIME_FRAMES = 720;
const MIN_DB = -90;
const MAX_DB = 0;

function isPositiveFinite(value) {
  return Number.isFinite(value) && value > 0;
}

function isPowerOfTwo(value) {
  return Number.isInteger(value) && value > 1 && (value & (value - 1)) === 0;
}

function isAudioBufferLike(input) {
  return Boolean(
    input
    && isPositiveFinite(input.sampleRate)
    && Number.isInteger(input.length)
    && input.length > 0
    && typeof input.getChannelData === "function"
  );
}

function validateFftOptions(options) {
  const fftSize = options.fftSize ?? DEFAULT_FFT_SIZE;
  const hopSize = options.hopSize ?? DEFAULT_HOP_SIZE;

  if (!isPowerOfTwo(fftSize)) {
    throw new TypeError("fftSize must be a power of two greater than one.");
  }
  if (!Number.isInteger(hopSize) || hopSize < 1) {
    throw new TypeError("hopSize must be a positive integer.");
  }

  return { fftSize, hopSize };
}

function createHannWindow(size) {
  const windowValues = new Float32Array(size);
  for (let index = 0; index < size; index += 1) {
    windowValues[index] = 0.5 * (1 - Math.cos((2 * Math.PI * index) / (size - 1)));
  }
  return windowValues;
}

function fftInPlace(real, imaginary) {
  const size = real.length;

  for (let index = 1, reversed = 0; index < size; index += 1) {
    let bit = size >> 1;
    while (reversed & bit) {
      reversed ^= bit;
      bit >>= 1;
    }
    reversed ^= bit;
    if (index < reversed) {
      [real[index], real[reversed]] = [real[reversed], real[index]];
      [imaginary[index], imaginary[reversed]] = [imaginary[reversed], imaginary[index]];
    }
  }

  for (let blockSize = 2; blockSize <= size; blockSize <<= 1) {
    const angle = (-2 * Math.PI) / blockSize;
    const stepReal = Math.cos(angle);
    const stepImaginary = Math.sin(angle);
    const half = blockSize >> 1;

    for (let start = 0; start < size; start += blockSize) {
      let rotationReal = 1;
      let rotationImaginary = 0;
      for (let offset = 0; offset < half; offset += 1) {
        const evenIndex = start + offset;
        const oddIndex = evenIndex + half;
        const oddReal = real[oddIndex] * rotationReal - imaginary[oddIndex] * rotationImaginary;
        const oddImaginary = real[oddIndex] * rotationImaginary + imaginary[oddIndex] * rotationReal;
        const evenReal = real[evenIndex];
        const evenImaginary = imaginary[evenIndex];

        real[evenIndex] = evenReal + oddReal;
        imaginary[evenIndex] = evenImaginary + oddImaginary;
        real[oddIndex] = evenReal - oddReal;
        imaginary[oddIndex] = evenImaginary - oddImaginary;

        const nextReal = rotationReal * stepReal - rotationImaginary * stepImaginary;
        rotationImaginary = rotationReal * stepImaginary + rotationImaginary * stepReal;
        rotationReal = nextReal;
      }
    }
  }
}

export function computeSpectrogram(audioBuffer, options = {}) {
  if (!isAudioBufferLike(audioBuffer)) {
    throw new TypeError("Decoded AudioBuffer-like data is required.");
  }

  const { fftSize, hopSize } = validateFftOptions(options);
  const samples = audioBuffer.getChannelData(0);
  if (!samples || typeof samples.length !== "number" || samples.length < 1) {
    throw new TypeError("Decoded audio must contain channel sample data.");
  }

  const rawFrameCount = Math.max(1, Math.floor(Math.max(0, samples.length - fftSize) / hopSize) + 1);
  const frameStride = Math.max(1, Math.ceil(rawFrameCount / MAX_TIME_FRAMES));
  const frameCount = Math.ceil(rawFrameCount / frameStride);
  const binCount = fftSize / 2;
  const cells = new Float32Array(frameCount * binCount);
  const hannWindow = createHannWindow(fftSize);
  const real = new Float32Array(fftSize);
  const imaginary = new Float32Array(fftSize);

  for (let frame = 0; frame < frameCount; frame += 1) {
    const sourceOffset = frame * hopSize * frameStride;
    for (let index = 0; index < fftSize; index += 1) {
      real[index] = (samples[sourceOffset + index] || 0) * hannWindow[index];
      imaginary[index] = 0;
    }

    fftInPlace(real, imaginary);
    for (let bin = 0; bin < binCount; bin += 1) {
      const magnitude = (4 * Math.hypot(real[bin], imaginary[bin])) / fftSize;
      const decibels = 20 * Math.log10(Math.max(magnitude, 10 ** (MIN_DB / 20)));
      cells[frame * binCount + bin] = Math.max(MIN_DB, Math.min(MAX_DB, decibels));
    }
  }

  return {
    binCount,
    cells,
    duration: audioBuffer.duration ?? audioBuffer.length / audioBuffer.sampleRate,
    fftSize,
    frameCount,
    frameStride,
    hopSize,
    nyquist: audioBuffer.sampleRate / 2,
    sampleRate: audioBuffer.sampleRate,
  };
}

function formatTime(seconds) {
  if (seconds === 0) return "0 s";
  if (seconds < 1) return `${Number(seconds.toFixed(2))} s`;
  return `${Number(seconds.toFixed(1))} s`;
}

function formatFrequency(hertz) {
  if (hertz >= 1000) return `${(hertz / 1000).toFixed(1)} kHz`;
  return `${Math.round(hertz)} Hz`;
}

function colorForDb(decibels) {
  const value = Math.max(0, Math.min(1, (decibels - MIN_DB) / (MAX_DB - MIN_DB)));
  const stops = [
    [7, 17, 24],
    [21, 71, 102],
    [20, 143, 139],
    [238, 184, 73],
    [247, 236, 220],
  ];
  const scaled = value * (stops.length - 1);
  const lowerIndex = Math.min(stops.length - 2, Math.floor(scaled));
  const amount = scaled - lowerIndex;
  const lower = stops[lowerIndex];
  const upper = stops[lowerIndex + 1];
  const rgb = lower.map((channel, index) => Math.round(channel + (upper[index] - channel) * amount));
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

function prepareCanvas(canvas, displayWidth, displayHeight, devicePixelRatio) {
  const width = Math.max(1, Math.round(displayWidth));
  const height = Math.max(1, Math.round(displayHeight));
  const ratio = Math.max(1, devicePixelRatio || 1);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;

  const context = canvas.getContext("2d");
  if (!context) throw new Error("A 2D canvas context is required.");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  return { context, height, width };
}

export function drawSpectrogram(canvas, spectrogram, options = {}) {
  if (!canvas || !spectrogram || !spectrogram.cells) {
    throw new TypeError("Canvas and spectrogram data are required.");
  }

  const displayWidth = options.displayWidth ?? canvas.clientWidth ?? canvas.width;
  const displayHeight = options.displayHeight ?? canvas.clientHeight ?? canvas.height;
  const { context, height, width } = prepareCanvas(
    canvas,
    displayWidth,
    displayHeight,
    options.devicePixelRatio ?? globalThis.devicePixelRatio ?? 1
  );
  const margins = { bottom: 34, left: 54, right: 92, top: 14 };
  const plotWidth = Math.max(1, width - margins.left - margins.right);
  const plotHeight = Math.max(1, height - margins.top - margins.bottom);
  const cellWidth = plotWidth / spectrogram.frameCount;
  const cellHeight = plotHeight / spectrogram.binCount;

  context.fillStyle = "#071118";
  context.fillRect(0, 0, width, height);
  for (let frame = 0; frame < spectrogram.frameCount; frame += 1) {
    for (let bin = 0; bin < spectrogram.binCount; bin += 1) {
      const x = margins.left + frame * cellWidth;
      const y = margins.top + plotHeight - (bin + 1) * cellHeight;
      context.fillStyle = colorForDb(spectrogram.cells[frame * spectrogram.binCount + bin]);
      context.fillRect(x, y, Math.max(1, cellWidth + 0.35), Math.max(1, cellHeight + 0.35));
    }
  }

  context.strokeStyle = "rgba(255, 255, 255, 0.35)";
  context.lineWidth = 1;
  context.strokeRect(margins.left, margins.top, plotWidth, plotHeight);
  context.fillStyle = "rgba(255, 255, 255, 0.82)";
  context.font = "12px Roboto, Arial, sans-serif";

  context.textAlign = "left";
  context.textBaseline = "top";
  context.fillText("0 s", margins.left, margins.top + plotHeight + 9);
  context.textAlign = "right";
  context.fillText(formatTime(spectrogram.duration), margins.left + plotWidth, margins.top + plotHeight + 9);

  context.textAlign = "right";
  context.textBaseline = "bottom";
  context.fillText("0 Hz", margins.left - 7, margins.top + plotHeight);
  context.textBaseline = "top";
  context.fillText(formatFrequency(spectrogram.nyquist), margins.left - 7, margins.top);

  const legendX = margins.left + plotWidth + 20;
  const legendWidth = 15;
  const legendSteps = Math.max(12, Math.round(plotHeight));
  for (let step = 0; step < legendSteps; step += 1) {
    const ratio = 1 - step / Math.max(1, legendSteps - 1);
    context.fillStyle = colorForDb(MIN_DB + ratio * (MAX_DB - MIN_DB));
    context.fillRect(legendX, margins.top + step * plotHeight / legendSteps, legendWidth, plotHeight / legendSteps + 1);
  }
  context.fillStyle = "rgba(255, 255, 255, 0.82)";
  context.textAlign = "left";
  context.textBaseline = "top";
  context.fillText("0 dB", legendX + 21, margins.top);
  context.textBaseline = "bottom";
  context.fillText(`${MIN_DB} dB`, legendX + 21, margins.top + plotHeight);
  context.save();
  context.textAlign = "center";
  context.textBaseline = "bottom";
  context.fillText("Intensity (dB)", legendX + 8, margins.top - 1);
  context.restore();

  canvas.setAttribute(
    "aria-label",
    `Spectrogram from 0 to ${formatTime(spectrogram.duration)}, 0 Hz to ${formatFrequency(spectrogram.nyquist)}.`
  );
}

export function decodeFloat32PcmBase64(encoded, sampleRate) {
  if (!isPositiveFinite(sampleRate)) {
    throw new TypeError("A positive sample rate is required.");
  }
  if (typeof encoded !== "string" || encoded.length === 0) {
    throw new TypeError("Valid base64 audio data is required.");
  }

  let binary;
  try {
    binary = atob(encoded);
  } catch (_error) {
    throw new TypeError("Valid base64 audio data is required.");
  }
  if (binary.length === 0 || binary.length % Float32Array.BYTES_PER_ELEMENT !== 0) {
    throw new TypeError("Valid float32 audio data is required.");
  }

  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
  const dataView = new DataView(bytes.buffer);
  const samples = new Float32Array(bytes.byteLength / Float32Array.BYTES_PER_ELEMENT);
  for (let index = 0; index < samples.length; index += 1) {
    samples[index] = dataView.getFloat32(index * Float32Array.BYTES_PER_ELEMENT, true);
    if (!Number.isFinite(samples[index])) {
      throw new TypeError("Valid float32 audio data is required.");
    }
  }

  return {
    duration: samples.length / sampleRate,
    getChannelData(channel) {
      if (channel !== 0) throw new RangeError("Only channel zero is available.");
      return samples;
    },
    length: samples.length,
    numberOfChannels: 1,
    sampleRate,
  };
}

function defaultCreateAudioContext() {
  const AudioContextClass = globalThis.AudioContext || globalThis.webkitAudioContext;
  if (!AudioContextClass) throw new Error("Web Audio is unavailable.");
  return new AudioContextClass();
}

export class AudioDecoder {
  constructor(options = {}) {
    this.createAudioContext = options.createAudioContext || defaultCreateAudioContext;
    this.contexts = new Set();
    this.closedContexts = new WeakSet();
    this.destroyed = false;
  }

  async closeContext(context) {
    if (!context || this.closedContexts.has(context)) return;
    this.closedContexts.add(context);
    this.contexts.delete(context);
    if (typeof context.close === "function") {
      try {
        await context.close();
      } catch (_error) {
        // Closing a browser-owned context is best effort during teardown.
      }
    }
  }

  async decode(input) {
    if (this.destroyed) throw new Error("Audio decoder has been destroyed.");
    if (isAudioBufferLike(input)) return input;

    let encodedData;
    if (input instanceof ArrayBuffer) {
      encodedData = input.slice(0);
    } else if (ArrayBuffer.isView(input)) {
      encodedData = input.buffer.slice(input.byteOffset, input.byteOffset + input.byteLength);
    } else if (input && typeof input.arrayBuffer === "function") {
      encodedData = await input.arrayBuffer();
    } else {
      throw new TypeError("ArrayBuffer, Blob, File, or decoded audio data is required.");
    }

    if (!(encodedData instanceof ArrayBuffer) || encodedData.byteLength === 0) {
      throw new TypeError("Encoded audio data is empty.");
    }

    const context = this.createAudioContext();
    this.contexts.add(context);
    try {
      if (!context || typeof context.decodeAudioData !== "function") {
        throw new Error("A compatible AudioContext is required.");
      }
      const decoded = await new Promise((resolve, reject) => {
        let decodeResult;
        try {
          decodeResult = context.decodeAudioData(encodedData.slice(0), resolve, reject);
        } catch (error) {
          reject(error);
          return;
        }
        if (decodeResult && typeof decodeResult.then === "function") {
          decodeResult.then(resolve, reject);
        }
      });
      if (!isAudioBufferLike(decoded)) {
        throw new Error("Browser audio decoding returned invalid data.");
      }
      return decoded;
    } finally {
      await this.closeContext(context);
    }
  }

  async destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    await Promise.allSettled(Array.from(this.contexts, (context) => this.closeContext(context)));
  }
}

function defaultRequestAnimationFrame(callback) {
  if (typeof globalThis.requestAnimationFrame === "function") {
    return globalThis.requestAnimationFrame(callback);
  }
  return globalThis.setTimeout(callback, 16);
}

function defaultCancelAnimationFrame(id) {
  if (typeof globalThis.cancelAnimationFrame === "function") {
    globalThis.cancelAnimationFrame(id);
  } else {
    globalThis.clearTimeout(id);
  }
}

export class SpectrogramView {
  constructor(root, options = {}) {
    if (!root || typeof root.querySelector !== "function") {
      throw new TypeError("A spectrogram root element is required.");
    }
    this.root = root;
    this.canvas = root.querySelector('[data-role="canvas"]');
    this.viewport = root.querySelector('[data-role="viewport"]');
    this.summary = root.querySelector('[data-role="summary"]');
    this.stateElements = {
      empty: root.querySelector('[data-role="empty"]'),
      error: root.querySelector('[data-role="error"]'),
      loading: root.querySelector('[data-role="loading"]'),
    };
    if (!this.canvas || !this.viewport) {
      throw new Error("Spectrogram canvas and viewport elements are required.");
    }

    this.decoder = options.decoder || new AudioDecoder(options);
    this.fftOptions = options.fftOptions || {};
    this.devicePixelRatio = options.devicePixelRatio ?? globalThis.devicePixelRatio ?? 1;
    this.requestAnimationFrame = options.requestAnimationFrame || defaultRequestAnimationFrame;
    this.cancelAnimationFrame = options.cancelAnimationFrame || defaultCancelAnimationFrame;
    this.destroyed = false;
    this.frameId = null;
    this.loadSequence = 0;
    this.result = null;

    const ResizeObserverClass = options.ResizeObserverClass || globalThis.ResizeObserver;
    this.resizeObserver = ResizeObserverClass ? new ResizeObserverClass(() => this.scheduleRender()) : null;
    if (this.resizeObserver) this.resizeObserver.observe(this.viewport);
    this.setState("empty");
  }

  setState(state, message = "") {
    this.root.dataset.state = state;
    for (const [name, element] of Object.entries(this.stateElements)) {
      if (element) element.hidden = name !== state;
    }
    if (state === "error" && this.stateElements.error) {
      this.stateElements.error.textContent = message;
    }
    if (state !== "success" && this.summary) this.summary.textContent = "";
  }

  showEmpty() {
    this.loadSequence += 1;
    this.result = null;
    this.setState("empty");
    this.scheduleRender();
  }

  showLoading() {
    this.loadSequence += 1;
    this.result = null;
    this.setState("loading");
    this.scheduleRender();
  }

  showDecodeError() {
    this.loadSequence += 1;
    this.result = null;
    this.setState("error", AUDIO_DECODE_ERROR_MESSAGE);
    this.scheduleRender();
  }

  showLoadError() {
    this.loadSequence += 1;
    this.result = null;
    this.setState("error", AUDIO_LOAD_ERROR_MESSAGE);
    this.scheduleRender();
  }

  async load(input) {
    if (this.destroyed) return null;
    if (input == null) {
      this.showEmpty();
      return null;
    }

    const sequence = ++this.loadSequence;
    this.result = null;
    this.setState("loading");
    this.scheduleRender();
    try {
      const decodedAudio = await this.decoder.decode(input);
      if (this.destroyed || sequence !== this.loadSequence) return null;
      const result = computeSpectrogram(decodedAudio, this.fftOptions);
      this.result = result;
      if (this.summary) {
        this.summary.textContent = `${formatTime(result.duration)} | ${formatFrequency(result.sampleRate)} sample rate | ${formatFrequency(result.nyquist)} Nyquist`;
      }
      this.setState("success");
      this.scheduleRender();
      return result;
    } catch (_error) {
      if (!this.destroyed && sequence === this.loadSequence) {
        this.result = null;
        this.setState("error", AUDIO_DECODE_ERROR_MESSAGE);
        this.scheduleRender();
      }
      return null;
    }
  }

  scheduleRender() {
    if (this.destroyed || this.frameId !== null) return;
    this.frameId = this.requestAnimationFrame(() => {
      this.frameId = null;
      this.render();
    });
  }

  render() {
    if (this.destroyed) return;
    const width = Math.max(1, this.viewport.clientWidth || 1);
    const height = Math.max(1, this.viewport.clientHeight || 1);
    if (this.result) {
      drawSpectrogram(this.canvas, this.result, {
        devicePixelRatio: this.devicePixelRatio,
        displayHeight: height,
        displayWidth: width,
      });
      return;
    }

    const { context } = prepareCanvas(this.canvas, width, height, this.devicePixelRatio);
    this.canvas.removeAttribute("aria-label");
    context.fillStyle = "#071118";
    context.fillRect(0, 0, width, height);
  }

  async destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    this.loadSequence += 1;
    if (this.resizeObserver) this.resizeObserver.disconnect();
    if (this.frameId !== null) {
      this.cancelAnimationFrame(this.frameId);
      this.frameId = null;
    }
    await this.decoder.destroy();
  }
}
