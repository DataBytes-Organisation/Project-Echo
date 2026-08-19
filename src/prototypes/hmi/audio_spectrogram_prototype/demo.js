(function createDemoModule(globalObject, factory) {
  "use strict";

  const api = factory();

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }

  if (globalObject && globalObject.document && globalObject.EchoSpectrogram) {
    api.initializeDemo(globalObject);
  }
}(typeof globalThis !== "undefined" ? globalThis : this, function demoFactory() {
  "use strict";

  function createPanelLoader(views, setStatus) {
    let sequence = 0;

    return {
      invalidate(message) {
        sequence += 1;
        setStatus(message);
      },

      async load(input, pendingMessage, successMessage) {
        const currentSequence = ++sequence;
        setStatus(pendingMessage);
        const results = await Promise.all(views.map((view) => view.load(input)));

        if (currentSequence === sequence) {
          setStatus(results.every(Boolean) ? successMessage : "Audio could not be decoded");
        }
        return results;
      },
    };
  }

  function createGeneratedAudio() {
    const sampleRate = 12000;
    const duration = 3;
    const length = sampleRate * duration;
    const samples = new Float32Array(length);

    for (let index = 0; index < length; index += 1) {
      const seconds = index / sampleRate;
      const pulse = seconds % 0.75 < 0.42 ? 1 : 0.18;
      const baseCall = Math.sin(2 * Math.PI * 760 * seconds);
      const overtone = Math.sin(2 * Math.PI * (1850 + 420 * seconds) * seconds);
      const lowBed = Math.sin(2 * Math.PI * 180 * seconds);
      samples[index] = pulse * (0.58 * baseCall + 0.24 * overtone) + 0.08 * lowBed;
    }

    return {
      duration,
      length,
      numberOfChannels: 1,
      sampleRate,
      getChannelData(channel) {
        if (channel !== 0) {
          throw new RangeError("Generated audio has one channel.");
        }
        return samples;
      },
    };
  }

  function initializeDemo(browserWindow) {
    const { document, EchoSpectrogram } = browserWindow;
    const { SpectrogramView } = EchoSpectrogram;
    const panelRoots = Array.from(document.querySelectorAll("[data-spectrogram-panel]"));
    const views = panelRoots.map((root) => new SpectrogramView(root, {
      fftOptions: { fftSize: 256, hopSize: 64 },
    }));
    const status = document.querySelector("[data-demo-status]");
    const fileInput = document.querySelector('[data-action="file"]');
    const loader = createPanelLoader(views, (message) => {
      status.textContent = message;
    });

    document.querySelector('[data-action="sample"]').addEventListener("click", () => {
      void loader.load(
        createGeneratedAudio(),
        "Rendering generated clip",
        "Generated clip | 3.00 s | 12.0 kHz"
      );
    });

    fileInput.addEventListener("change", () => {
      const [file] = fileInput.files;
      if (!file) {
        return;
      }
      void loader.load(file, `Decoding ${file.name}`, `Local clip | ${file.name}`);
    });

    document.querySelector('[data-action="empty"]').addEventListener("click", () => {
      views.forEach((view) => view.showEmpty());
      loader.invalidate("No clip selected");
      fileInput.value = "";
    });

    document.querySelector('[data-action="loading"]').addEventListener("click", () => {
      views.forEach((view) => view.showLoading());
      loader.invalidate("Decoding audio");
    });

    document.querySelector('[data-action="error"]').addEventListener("click", () => {
      const invalidAudio = new Uint8Array([78, 79, 84, 65, 85, 68, 73, 79]).buffer;
      void loader.load(invalidAudio, "Testing audio decoder", "Unexpected decode success");
    });

    browserWindow.addEventListener("beforeunload", () => {
      for (const view of views) {
        void view.destroy();
      }
    }, { once: true });
  }

  return {
    createPanelLoader,
    initializeDemo,
  };
}));
