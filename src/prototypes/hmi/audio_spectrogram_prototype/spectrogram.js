(function createSpectrogramModule(globalObject, factory) {
  "use strict";

  const api = factory(globalObject);

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }

  if (globalObject) {
    globalObject.EchoSpectrogram = api;
  }
}(typeof globalThis !== "undefined" ? globalThis : this, function spectrogramFactory(globalObject) {
  "use strict";

  const AUDIO_DECODE_ERROR_MESSAGE =
    "We couldn't decode this audio clip. Choose a supported audio file and try again.";
  const DEFAULT_FFT_SIZE = 256;
  const DEFAULT_HOP_SIZE = 64;
  const DEFAULT_MAX_FRAMES = 720;
  const MIN_DB = -100;
  const MAX_DB = 0;

  function isPositiveFinite(value) {
    return Number.isFinite(value) && value > 0;
  }

  function isPowerOfTwo(value) {
    return Number.isInteger(value) && value >= 32 && (value & (value - 1)) === 0;
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
    const maxFrames = options.maxFrames ?? DEFAULT_MAX_FRAMES;

    if (!isPowerOfTwo(fftSize)) {
      throw new RangeError("fftSize must be a power of two and at least 32.");
    }
    if (!Number.isInteger(hopSize) || hopSize < 1 || hopSize > fftSize) {
      throw new RangeError("hopSize must be an integer between 1 and fftSize.");
    }
    if (!Number.isInteger(maxFrames) || maxFrames < 1) {
      throw new RangeError("maxFrames must be a positive integer.");
    }

    return { fftSize, hopSize, maxFrames };
  }

  function createHannWindow(size) {
    const windowValues = new Float64Array(size);
    for (let index = 0; index < size; index += 1) {
      windowValues[index] = 0.5 * (1 - Math.cos(2 * Math.PI * index / (size - 1)));
    }
    return windowValues;
  }

  function fftInPlace(real, imaginary) {
    const size = real.length;
    let reversed = 0;

    for (let index = 1; index < size; index += 1) {
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

    for (let width = 2; width <= size; width <<= 1) {
      const angle = -2 * Math.PI / width;
      const stepReal = Math.cos(angle);
      const stepImaginary = Math.sin(angle);

      for (let offset = 0; offset < size; offset += width) {
        let twiddleReal = 1;
        let twiddleImaginary = 0;
        const half = width >> 1;

        for (let index = 0; index < half; index += 1) {
          const evenIndex = offset + index;
          const oddIndex = evenIndex + half;
          const oddReal = real[oddIndex] * twiddleReal
            - imaginary[oddIndex] * twiddleImaginary;
          const oddImaginary = real[oddIndex] * twiddleImaginary
            + imaginary[oddIndex] * twiddleReal;
          const evenReal = real[evenIndex];
          const evenImaginary = imaginary[evenIndex];

          real[evenIndex] = evenReal + oddReal;
          imaginary[evenIndex] = evenImaginary + oddImaginary;
          real[oddIndex] = evenReal - oddReal;
          imaginary[oddIndex] = evenImaginary - oddImaginary;

          const nextTwiddleReal = twiddleReal * stepReal
            - twiddleImaginary * stepImaginary;
          twiddleImaginary = twiddleReal * stepImaginary
            + twiddleImaginary * stepReal;
          twiddleReal = nextTwiddleReal;
        }
      }
    }
  }

  function computeSpectrogram(audioBuffer, options = {}) {
    if (!isAudioBufferLike(audioBuffer)) {
      throw new TypeError("A valid decoded AudioBuffer-like value is required.");
    }

    const { fftSize, hopSize, maxFrames } = validateFftOptions(options);
    const samples = audioBuffer.getChannelData(0);
    if (!samples || samples.length < 1) {
      throw new TypeError("Decoded audio channel zero is empty.");
    }

    const sampleRate = audioBuffer.sampleRate;
    const duration = isPositiveFinite(audioBuffer.duration)
      ? audioBuffer.duration
      : samples.length / sampleRate;
    const binCount = fftSize / 2;
    const sampleSpan = Math.max(0, samples.length - fftSize);
    const effectiveHopSize = maxFrames === 1
      ? Math.max(1, samples.length)
      : Math.max(hopSize, Math.ceil(sampleSpan / (maxFrames - 1)));
    const frameCount = sampleSpan === 0 || maxFrames === 1
      ? 1
      : 1 + Math.ceil(sampleSpan / effectiveHopSize);
    const cells = new Float32Array(frameCount * binCount);
    const hannWindow = createHannWindow(fftSize);
    const real = new Float64Array(fftSize);
    const imaginary = new Float64Array(fftSize);
    const magnitudeScale = 4 / fftSize;

    for (let frame = 0; frame < frameCount; frame += 1) {
      const sampleOffset = frame * effectiveHopSize;

      for (let index = 0; index < fftSize; index += 1) {
        real[index] = (samples[sampleOffset + index] || 0) * hannWindow[index];
        imaginary[index] = 0;
      }

      fftInPlace(real, imaginary);

      for (let bin = 0; bin < binCount; bin += 1) {
        const magnitude = Math.hypot(real[bin], imaginary[bin]) * magnitudeScale;
        const decibels = 20 * Math.log10(Math.max(magnitude, 1e-10));
        cells[frame * binCount + bin] = Math.max(MIN_DB, Math.min(MAX_DB, decibels));
      }
    }

    return {
      binCount,
      cells,
      duration,
      fftSize,
      frameCount,
      hopSize: effectiveHopSize,
      maxDb: MAX_DB,
      minDb: MIN_DB,
      nyquist: sampleRate / 2,
      sampleRate,
    };
  }

  function formatTime(seconds) {
    if (Math.abs(seconds) < 0.0005) {
      return "0 s";
    }
    const precision = seconds < 10 ? 2 : 1;
    return `${seconds.toFixed(precision).replace(/\.0+$|(?<=\.[0-9])0+$/, "")} s`;
  }

  function formatFrequency(hertz) {
    if (hertz >= 1000) {
      return `${(hertz / 1000).toFixed(1)} kHz`;
    }
    return `${Math.round(hertz)} Hz`;
  }

  function colorForDb(decibels) {
    const normalized = Math.max(0, Math.min(1, (decibels - MIN_DB) / (MAX_DB - MIN_DB)));
    const stops = [
      [7, 17, 24],
      [15, 67, 71],
      [38, 104, 126],
      [213, 177, 75],
      [229, 96, 57],
      [255, 241, 196],
    ];
    const position = normalized * (stops.length - 1);
    const lowerIndex = Math.min(stops.length - 2, Math.floor(position));
    const mix = position - lowerIndex;
    const lower = stops[lowerIndex];
    const upper = stops[lowerIndex + 1];
    const channels = lower.map((channel, index) => Math.round(
      channel + (upper[index] - channel) * mix
    ));
    return `rgb(${channels[0]}, ${channels[1]}, ${channels[2]})`;
  }

  function prepareCanvas(canvas, displayWidth, displayHeight, devicePixelRatio) {
    const width = Math.max(1, Math.round(displayWidth));
    const height = Math.max(1, Math.round(displayHeight));
    const ratio = Math.max(1, Number(devicePixelRatio) || 1);
    const backingWidth = Math.max(1, Math.round(width * ratio));
    const backingHeight = Math.max(1, Math.round(height * ratio));

    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    if (canvas.width !== backingWidth) {
      canvas.width = backingWidth;
    }
    if (canvas.height !== backingHeight) {
      canvas.height = backingHeight;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      throw new Error("A Canvas 2D context is required.");
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    return { context, height, ratio, width };
  }

  function drawSpectrogram(canvas, spectrogram, options = {}) {
    if (!canvas || typeof canvas.getContext !== "function") {
      throw new TypeError("A canvas element is required.");
    }
    if (!spectrogram || !spectrogram.cells || spectrogram.cells.length < 1) {
      throw new TypeError("Non-empty spectrogram data is required.");
    }

    const ratio = options.devicePixelRatio ?? globalObject.devicePixelRatio ?? 1;
    const displayWidth = options.displayWidth
      ?? canvas.clientWidth
      ?? canvas.width / ratio
      ?? 1;
    const displayHeight = options.displayHeight
      ?? canvas.clientHeight
      ?? canvas.height / ratio
      ?? 1;
    const { context, height, width } = prepareCanvas(
      canvas,
      displayWidth,
      displayHeight,
      ratio
    );
    const margin = { bottom: 34, left: 54, right: 72, top: 16 };
    const plotWidth = Math.max(1, width - margin.left - margin.right);
    const plotHeight = Math.max(1, height - margin.top - margin.bottom);
    const cellWidth = plotWidth / spectrogram.frameCount;
    const cellHeight = plotHeight / spectrogram.binCount;

    context.fillStyle = "#071118";
    context.fillRect(0, 0, width, height);

    for (let frame = 0; frame < spectrogram.frameCount; frame += 1) {
      for (let bin = 0; bin < spectrogram.binCount; bin += 1) {
        const decibels = spectrogram.cells[frame * spectrogram.binCount + bin];
        const x = margin.left + frame * cellWidth;
        const y = margin.top + (spectrogram.binCount - bin - 1) * cellHeight;
        context.fillStyle = colorForDb(decibels);
        context.fillRect(x, y, Math.max(1, cellWidth + 0.25), Math.max(1, cellHeight + 0.25));
      }
    }

    context.strokeStyle = "#6f858b";
    context.lineWidth = 1;
    context.strokeRect(margin.left, margin.top, plotWidth, plotHeight);
    context.fillStyle = "#c7d5d5";
    context.font = "12px system-ui, sans-serif";

    const timeTicks = [0, 0.5, 1];
    context.textBaseline = "top";
    for (const tick of timeTicks) {
      const x = margin.left + tick * plotWidth;
      context.textAlign = tick === 0 ? "left" : tick === 1 ? "right" : "center";
      context.fillText(formatTime(spectrogram.duration * tick), x, margin.top + plotHeight + 8);
    }

    const frequencyTicks = [0, 0.5, 1];
    context.textAlign = "right";
    context.textBaseline = "middle";
    for (const tick of frequencyTicks) {
      const y = margin.top + plotHeight - tick * plotHeight;
      context.fillText(formatFrequency(spectrogram.nyquist * tick), margin.left - 8, y);
    }

    const legendX = width - 48;
    const legendWidth = 12;
    const legendSteps = 24;
    for (let step = 0; step < legendSteps; step += 1) {
      const normalized = step / (legendSteps - 1);
      context.fillStyle = colorForDb(MIN_DB + normalized * (MAX_DB - MIN_DB));
      context.fillRect(
        legendX,
        margin.top + plotHeight - (step + 1) * plotHeight / legendSteps,
        legendWidth,
        plotHeight / legendSteps + 1
      );
    }

    context.fillStyle = "#c7d5d5";
    context.font = "11px system-ui, sans-serif";
    context.textAlign = "left";
    context.textBaseline = "top";
    context.fillText("0 dB", legendX + 17, margin.top - 1);
    context.textBaseline = "bottom";
    context.fillText("-100", legendX + 17, margin.top + plotHeight + 1);
    context.save();
    context.textAlign = "center";
    context.textBaseline = "bottom";
    context.fillText("Intensity (dB)", legendX + 6, margin.top + plotHeight + 30);
    context.restore();

    if (typeof canvas.setAttribute === "function") {
      canvas.setAttribute(
        "aria-label",
        `Spectrogram from 0 to ${formatTime(spectrogram.duration)}, 0 Hz to ${formatFrequency(spectrogram.nyquist)}, intensity ${MIN_DB} to ${MAX_DB} dB.`
      );
    }

    return { height, plotHeight, plotWidth, width };
  }

  async function inputToArrayBuffer(input) {
    if (input instanceof ArrayBuffer) {
      return input;
    }
    if (ArrayBuffer.isView(input)) {
      return input.buffer.slice(input.byteOffset, input.byteOffset + input.byteLength);
    }
    if (input && typeof input.arrayBuffer === "function") {
      const value = await input.arrayBuffer();
      if (value instanceof ArrayBuffer) {
        return value;
      }
    }
    throw new TypeError("Use decoded audio, an ArrayBuffer, Blob, or File-shaped input.");
  }

  function defaultCreateAudioContext() {
    const AudioContextClass = globalObject.AudioContext || globalObject.webkitAudioContext;
    if (!AudioContextClass) {
      throw new Error("Browser audio decoding is unavailable.");
    }
    return new AudioContextClass();
  }

  class AudioDecoder {
    constructor(options = {}) {
      this.createAudioContext = options.createAudioContext || defaultCreateAudioContext;
      this.contexts = new Set();
      this.destroyed = false;
    }

    async releaseContext(context) {
      if (!this.contexts.has(context)) {
        return;
      }
      this.contexts.delete(context);
      if (typeof context.close === "function") {
        try {
          await context.close();
        } catch (_error) {
          // Context shutdown must not replace the decode result or decode error.
        }
      }
    }

    async decode(input) {
      if (this.destroyed) {
        throw new Error("The audio decoder has been destroyed.");
      }
      if (isAudioBufferLike(input)) {
        return input;
      }

      const encodedAudio = await inputToArrayBuffer(input);
      if (this.destroyed) {
        throw new Error("The audio decoder has been destroyed.");
      }

      const context = this.createAudioContext();
      if (!context) {
        throw new Error("A compatible AudioContext is required.");
      }
      this.contexts.add(context);

      try {
        if (typeof context.decodeAudioData !== "function") {
          throw new Error("A compatible AudioContext is required.");
        }
        const decoded = await context.decodeAudioData(encodedAudio.slice(0));
        if (!isAudioBufferLike(decoded)) {
          throw new Error("The browser returned invalid decoded audio.");
        }
        return decoded;
      } finally {
        await this.releaseContext(context);
      }
    }

    async destroy() {
      if (this.destroyed && this.contexts.size === 0) {
        return;
      }
      this.destroyed = true;
      await Promise.all(Array.from(this.contexts, (context) => this.releaseContext(context)));
    }
  }

  function defaultRequestAnimationFrame(callback) {
    if (typeof globalObject.requestAnimationFrame === "function") {
      return globalObject.requestAnimationFrame(callback);
    }
    return globalObject.setTimeout(callback, 16);
  }

  function defaultCancelAnimationFrame(id) {
    if (typeof globalObject.cancelAnimationFrame === "function") {
      globalObject.cancelAnimationFrame(id);
      return;
    }
    globalObject.clearTimeout(id);
  }

  class SpectrogramView {
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
      this.devicePixelRatio = options.devicePixelRatio
        ?? globalObject.devicePixelRatio
        ?? 1;
      this.requestAnimationFrame = options.requestAnimationFrame
        || defaultRequestAnimationFrame;
      this.cancelAnimationFrame = options.cancelAnimationFrame
        || defaultCancelAnimationFrame;
      this.destroyed = false;
      this.frameId = null;
      this.loadSequence = 0;
      this.result = null;

      const ResizeObserverClass = options.ResizeObserverClass || globalObject.ResizeObserver;
      this.resizeObserver = ResizeObserverClass
        ? new ResizeObserverClass(() => this.scheduleRender())
        : null;
      if (this.resizeObserver) {
        this.resizeObserver.observe(this.viewport);
      }

      this.setState("empty");
    }

    setState(state, message = "") {
      this.root.dataset.state = state;
      for (const [name, element] of Object.entries(this.stateElements)) {
        if (element) {
          element.hidden = name !== state;
        }
      }
      if (state === "error" && this.stateElements.error) {
        this.stateElements.error.textContent = message;
      }
      if (state !== "success" && this.summary) {
        this.summary.textContent = "";
      }
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

    async load(input) {
      if (this.destroyed) {
        return null;
      }
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
        if (this.destroyed || sequence !== this.loadSequence) {
          return null;
        }

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
      if (this.destroyed || this.frameId !== null) {
        return;
      }
      this.frameId = this.requestAnimationFrame(() => {
        this.frameId = null;
        this.render();
      });
    }

    render() {
      if (this.destroyed) {
        return;
      }
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

      const { context } = prepareCanvas(
        this.canvas,
        width,
        height,
        this.devicePixelRatio
      );
      if (typeof this.canvas.removeAttribute === "function") {
        this.canvas.removeAttribute("aria-label");
      }
      context.fillStyle = "#071118";
      context.fillRect(0, 0, width, height);
    }

    async destroy() {
      if (this.destroyed) {
        return;
      }
      this.destroyed = true;
      this.loadSequence += 1;
      if (this.resizeObserver) {
        this.resizeObserver.disconnect();
      }
      if (this.frameId !== null) {
        this.cancelAnimationFrame(this.frameId);
        this.frameId = null;
      }
      await this.decoder.destroy();
    }
  }

  return {
    AUDIO_DECODE_ERROR_MESSAGE,
    AudioDecoder,
    SpectrogramView,
    computeSpectrogram,
    drawSpectrogram,
  };
}));
