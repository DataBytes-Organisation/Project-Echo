//API to handle audio recording 
export function getAudioRecorder(options = {}){

const browserGetUserMedia = globalThis.navigator?.mediaDevices?.getUserMedia
    ? globalThis.navigator.mediaDevices.getUserMedia.bind(globalThis.navigator.mediaDevices)
    : null;
const getUserMedia = options.getUserMedia || browserGetUserMedia;
const MediaRecorderClass = options.MediaRecorderClass || globalThis.MediaRecorder;

export const MAX_RECORDING_SECONDS = 20;

/** Keep only the first maxSeconds of samples. */
export function clampSamplesToMax(samples, sampleRate, maxSeconds = MAX_RECORDING_SECONDS) {
  if (!samples || samples.length === 0) return samples || new Float32Array(0);
  const rate = sampleRate || 44100;
  const maxSamples = Math.floor(rate * maxSeconds);
  if (samples.length <= maxSamples) return samples;
  return samples.slice(0, maxSamples);
}

/** Encode mono Float32 PCM samples as a 16-bit WAV blob. */
export function encodeWavBlob(samples, sampleRate) {
  const numSamples = samples.length;
  const dataSize = numSamples * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  const writeString = (offset, value) => {
    for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i));
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < numSamples; i++, offset += 2) {
    let sample = samples[i];
    if (sample > 1) sample = 1;
    else if (sample < -1) sample = -1;
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }

  return new Blob([new Uint8Array(buffer)], { type: "audio/wav" });
}

/**
 * Build a capped WAV recording from PCM samples and/or MediaRecorder chunks.
 * Both paths hard-clamp to MAX_RECORDING_SECONDS before returning.
 *
 * @param {object} options
 * @param {Float32Array} options.pcmSamples
 * @param {number} options.sampleRate
 * @param {Blob[]} [options.mediaBlobs]
 * @param {string} [options.mediaMimeType]
 * @param {(blob: Blob) => Promise<{samples: Float32Array, sampleRate: number}>} [options.decodeAudioBlob]
 */
export async function buildCappedRecording({
  pcmSamples,
  sampleRate,
  mediaBlobs = [],
  mediaMimeType = "audio/webm",
  decodeAudioBlob,
}) {
  const rate = sampleRate || 44100;
  let cappedSamples = clampSamplesToMax(pcmSamples, rate);

  if (cappedSamples && cappedSamples.length > 0) {
    return {
      blob: encodeWavBlob(cappedSamples, rate),
      samples: cappedSamples,
      sampleRate: rate,
      mimeType: "audio/wav",
    };
  }

  if (!mediaBlobs || mediaBlobs.length === 0) {
    throw new Error("No audio was captured");
  }

  if (typeof decodeAudioBlob !== "function") {
    throw new Error("Could not process the recorded audio. Please try recording again.");
  }

  const rawBlob = new Blob(mediaBlobs, { type: mediaMimeType || "audio/webm" });
  let decoded;
  try {
    decoded = await decodeAudioBlob(rawBlob);
  } catch (_err) {
    throw new Error("Could not process the recorded audio. Please try recording again.");
  }

  if (!decoded || !decoded.samples || decoded.samples.length === 0) {
    throw new Error("No audio was captured");
  }

  const decodedRate = decoded.sampleRate || rate;
  cappedSamples = clampSamplesToMax(decoded.samples, decodedRate);
  if (!cappedSamples.length) {
    throw new Error("No audio was captured");
  }

  return {
    blob: encodeWavBlob(cappedSamples, decodedRate),
    samples: cappedSamples,
    sampleRate: decodedRate,
    mimeType: "audio/wav",
  };
}

/** Decode a MediaRecorder blob to mono Float32 samples via Web Audio. */
export function decodeAudioBlobWithWebAudio(blob) {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) {
    return Promise.reject(new Error("Web Audio API is not available"));
  }

  const ctx = new AudioCtx();
  return blob
    .arrayBuffer()
    .then((arrayBuffer) => ctx.decodeAudioData(arrayBuffer))
    .then((audioBuffer) => {
      const channel = audioBuffer.getChannelData(0);
      // Copy before closing the context — channel views can be invalidated.
      const samples = new Float32Array(channel.length);
      samples.set(channel);
      return { samples, sampleRate: audioBuffer.sampleRate };
    })
    .finally(() => {
      if (ctx.state !== "closed") {
        return ctx.close().catch(() => {});
      }
    });
}

export function getAudioRecorder() {
  const audioRecorder = {
    audioBlobs: [],
    streamBeingCaptured: null,
    discardChunks: false,
    isRecording: false,
    isStarting: false,
    audioContext: null,
    scriptProcessor: null,
    mediaStreamSource: null,
    muteGain: null,
    mediaRecorder: null,
    pcmChunks: [],
    ownedStreams: new Set(),
    sampleRate: 44100,
    recorderMimeType: "audio/webm",

    start: function () {
        if (!getUserMedia || !MediaRecorderClass) {
            return Promise.reject(new Error('mediaDevices API or getUserMedia method is not supported in this browser.'));
        }

        else {
            
            return getUserMedia({ audio: true })
                .then(stream => {
                    audioRecorder.streamBeingCaptured = stream;
                    try {
                        audioRecorder.mediaRecorder = new MediaRecorderClass(stream);
                        audioRecorder.audioBlobs = [];
                        audioRecorder.mediaRecorder.addEventListener("dataavailable", event => {
                            audioRecorder.audioBlobs.push(event.data);
                        });
                        audioRecorder.mediaRecorder.start();
                    } catch (error) {
                        audioRecorder.stopStream();
                        audioRecorder.resetRecordingProperties();
                        throw error;
                    }
                });
        }
    },

      return navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
        audioRecorder.ownedStreams.add(stream);
        audioRecorder.streamBeingCaptured = stream;
        audioRecorder.audioBlobs = [];
        audioRecorder.pcmChunks = [];
        audioRecorder.discardChunks = false;

        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        const ctx = new AudioCtx();
        audioRecorder.audioContext = ctx;
        audioRecorder.sampleRate = ctx.sampleRate;

        const setupPcmGraph = () => {
          audioRecorder.mediaStreamSource = ctx.createMediaStreamSource(stream);
          audioRecorder.scriptProcessor = ctx.createScriptProcessor(4096, 1, 1);
          audioRecorder.scriptProcessor.onaudioprocess = (event) => {
            if (!audioRecorder.isRecording || audioRecorder.discardChunks) return;
            audioRecorder.pcmChunks.push(
              new Float32Array(event.inputBuffer.getChannelData(0))
            );
          };
          audioRecorder.muteGain = ctx.createGain();
          audioRecorder.muteGain.gain.value = 0;
          audioRecorder.mediaStreamSource.connect(audioRecorder.scriptProcessor);
          audioRecorder.scriptProcessor.connect(audioRecorder.muteGain);
          audioRecorder.muteGain.connect(ctx.destination);
        };

        const preferredTypes = [
          "audio/webm;codecs=opus",
          "audio/webm",
          "audio/ogg;codecs=opus",
          "audio/mp4",
        ];
        let mimeType = "";
        for (let i = 0; i < preferredTypes.length; i++) {
          if (window.MediaRecorder && MediaRecorder.isTypeSupported(preferredTypes[i])) {
            mimeType = preferredTypes[i];
            break;
          }
        }
        audioRecorder.recorderMimeType = mimeType || "audio/webm";

        try {
          audioRecorder.mediaRecorder = mimeType
            ? new MediaRecorder(stream, { mimeType })
            : new MediaRecorder(stream);
          audioRecorder.mediaRecorder.addEventListener("dataavailable", (event) => {
            if (audioRecorder.discardChunks) return;
            if (event.data && event.data.size > 0) {
              audioRecorder.audioBlobs.push(event.data);
            }
          });
          audioRecorder.mediaRecorder.start(200);
        } catch (_err) {
          audioRecorder.mediaRecorder = null;
        }

        const finishStart = () => {
          setupPcmGraph();
          audioRecorder.isRecording = true;
          audioRecorder.isStarting = false;
        };

        if (ctx.state === "suspended") {
          return ctx.resume().then(finishStart);
        }
        finishStart();
      }).catch((err) => {
        audioRecorder.isStarting = false;
        audioRecorder.isRecording = false;
        audioRecorder.cleanupGraph();
        audioRecorder.stopAllOwnedStreams();
        audioRecorder.resetRecordingProperties();
        throw err;
      });
    },

    clampSamplesToMax: clampSamplesToMax,
    encodeWavBlob: encodeWavBlob,

    mergePcmChunks: function () {
      const chunks = audioRecorder.pcmChunks;
      let length = 0;
      for (let i = 0; i < chunks.length; i++) length += chunks[i].length;
      const merged = new Float32Array(length);
      let offset = 0;
      for (let i = 0; i < chunks.length; i++) {
        merged.set(chunks[i], offset);
        offset += chunks[i].length;
      }
      return merged;
    },

    cleanupGraph: function () {
      try {
        if (audioRecorder.scriptProcessor) {
          audioRecorder.scriptProcessor.onaudioprocess = null;
          audioRecorder.scriptProcessor.disconnect();
        }
      } catch (_err) { /* ignore */ }
      try {
        if (audioRecorder.mediaStreamSource) audioRecorder.mediaStreamSource.disconnect();
      } catch (_err) { /* ignore */ }
      try {
        if (audioRecorder.muteGain) audioRecorder.muteGain.disconnect();
      } catch (_err) { /* ignore */ }
      if (audioRecorder.audioContext && audioRecorder.audioContext.state !== "closed") {
        audioRecorder.audioContext.close().catch(() => {});
      }
      audioRecorder.scriptProcessor = null;
      audioRecorder.mediaStreamSource = null;
      audioRecorder.muteGain = null;
      audioRecorder.audioContext = null;
    },

    stop: function () {
        return new Promise((resolve, reject) => {
            const recorder = audioRecorder.mediaRecorder;
            if (!recorder) {
                reject(new Error("No audio recording is active."));
                return;
            }
            const mimeType = recorder.mimeType || audioRecorder.audioBlobs[0]?.type || "audio/webm";
            recorder.addEventListener("stop", () => {
                resolve(new Blob(audioRecorder.audioBlobs, { type: mimeType }));
            }, { once: true });
            recorder.stop();
            audioRecorder.stopStream();
            audioRecorder.resetRecordingProperties();
        });
    },

    cancel: function () {
        if (audioRecorder.mediaRecorder && audioRecorder.mediaRecorder.state !== "inactive") {
            audioRecorder.mediaRecorder.stop();
        }
        audioRecorder.stopStream();
        audioRecorder.resetRecordingProperties();
        audioRecorder.audioBlobs = [];
    },

    stopStream: function () {
        if (audioRecorder.streamBeingCaptured) {
            audioRecorder.streamBeingCaptured.getTracks()
                .forEach(track => track.stop());
        }
    },

    resetRecordingProperties: function () {
      audioRecorder.streamBeingCaptured = null;
      audioRecorder.mediaRecorder = null;
      audioRecorder.isStarting = false;
    },
  };

  return audioRecorder;
}

return audioRecorder;

}
