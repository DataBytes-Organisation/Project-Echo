"use strict";

function requireFunction(value, name) {
  if (typeof value !== "function") {
    throw new TypeError(`${name} must be a function.`);
  }
  return value;
}

function requireView(view) {
  if (!view || typeof view.load !== "function") {
    throw new TypeError("A spectrogram view is required.");
  }
  return view;
}

export class LatestSourceGuard {
  constructor() {
    this.sequence = 0;
  }

  begin() {
    this.sequence += 1;
    return this.sequence;
  }

  invalidate() {
    this.sequence += 1;
  }

  isCurrent(token) {
    return token === this.sequence;
  }
}

export class AnimalSpectrogramWorkflow {
  constructor({ decodePcm, retrieveAudio, view }) {
    this.decodePcm = requireFunction(decodePcm, "decodePcm");
    this.retrieveAudio = requireFunction(retrieveAudio, "retrieveAudio");
    this.view = requireView(view);
    this.destroyed = false;
    this.loadPromise = null;
    this.loadSequence = 0;
    this.selectedAudio = null;
    this.selectedEventId = null;
  }

  async select(eventId) {
    if (this.destroyed) return null;
    if (eventId == null || eventId === "") {
      this.clear();
      return null;
    }

    const sequence = ++this.loadSequence;
    this.selectedAudio = null;
    this.selectedEventId = eventId;
    this.view.showLoading();

    const loadPromise = this.retrieveAudio(eventId)
      .then(async (response) => {
        if (this.destroyed || sequence !== this.loadSequence) return null;

        let decoded;
        try {
          decoded = this.decodePcm(response?.data?.audioClip, response?.data?.sampleRate);
        } catch (_error) {
          if (!this.destroyed && sequence === this.loadSequence) {
            this.view.showDecodeError();
          }
          return null;
        }

        if (this.destroyed || sequence !== this.loadSequence) return null;
        const renderResult = await this.view.load(decoded);
        if (this.destroyed || sequence !== this.loadSequence || renderResult === null) return null;
        this.selectedAudio = decoded;
        return decoded;
      })
      .catch(() => {
        if (!this.destroyed && sequence === this.loadSequence) {
          this.view.showLoadError();
        }
        return null;
      });

    this.loadPromise = loadPromise;
    return loadPromise;
  }

  async getSelectedAudio() {
    if (this.selectedAudio) return this.selectedAudio;
    if (!this.loadPromise) return null;
    return this.loadPromise;
  }

  clear() {
    if (this.destroyed) return;
    this.loadSequence += 1;
    this.loadPromise = null;
    this.selectedAudio = null;
    this.selectedEventId = null;
    this.view.showEmpty();
  }

  async destroy() {
    if (this.destroyed) return;
    this.loadSequence += 1;
    this.destroyed = true;
    this.loadPromise = null;
    this.selectedAudio = null;
    this.selectedEventId = null;
    await this.view.destroy();
  }
}

export class MicrophoneSpectrogramWorkflow {
  constructor({ decoder, view }) {
    if (!decoder || typeof decoder.decode !== "function") {
      throw new TypeError("An audio decoder is required.");
    }
    this.decoder = decoder;
    this.view = requireView(view);
    this.destroyed = false;
    this.loadPromise = null;
    this.loadSequence = 0;
    this.selectedAudio = null;
  }

  async load(input) {
    if (this.destroyed) return null;
    if (input == null) {
      this.clear();
      return null;
    }

    const sequence = ++this.loadSequence;
    this.selectedAudio = null;
    this.view.showLoading();
    const loadPromise = Promise.resolve()
      .then(() => this.decoder.decode(input))
      .then(async (decoded) => {
        if (this.destroyed || sequence !== this.loadSequence) return null;
        const renderResult = await this.view.load(decoded);
        if (this.destroyed || sequence !== this.loadSequence || renderResult === null) return null;
        this.selectedAudio = decoded;
        return decoded;
      })
      .catch(() => {
        if (!this.destroyed && sequence === this.loadSequence) {
          this.view.showDecodeError();
        }
        return null;
      });

    this.loadPromise = loadPromise;
    return loadPromise;
  }

  async getSelectedAudio() {
    if (this.selectedAudio) return this.selectedAudio;
    if (!this.loadPromise) return null;
    return this.loadPromise;
  }

  clear() {
    if (this.destroyed) return;
    this.loadSequence += 1;
    this.loadPromise = null;
    this.selectedAudio = null;
    this.view.showEmpty();
  }

  async destroy() {
    if (this.destroyed) return;
    this.loadSequence += 1;
    this.destroyed = true;
    this.loadPromise = null;
    this.selectedAudio = null;
    await Promise.allSettled([
      this.decoder.destroy(),
      this.view.destroy(),
    ]);
  }
}
