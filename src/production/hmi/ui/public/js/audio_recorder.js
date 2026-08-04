/**
 * Browser audio recording helper (FR-B2).
 * Captures:
 *  - MediaRecorder blob for native <audio controls> playback
 *  - PCM samples as backup / duration integrity
 * stop() returns { blob, samples, sampleRate, mimeType }
 */
export function getAudioRecorder() {
  const MAX_RECORDING_SECONDS = 20;

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
      if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
        return Promise.reject(
          new Error(
            "mediaDevices API or getUserMedia method is not supported in this browser."
          )
        );
      }

      if (audioRecorder.isRecording || audioRecorder.isStarting) {
        return Promise.reject(new Error("Recording already in progress"));
      }

      // Lock immediately to prevent concurrent getUserMedia() calls.
      audioRecorder.isStarting = true;

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

    clampSamplesToMax: function (samples, sampleRate) {
      if (!samples || samples.length === 0) return samples;
      const maxSamples = Math.floor(sampleRate * MAX_RECORDING_SECONDS);
      if (samples.length <= maxSamples) return samples;
      return samples.slice(0, maxSamples);
    },

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

    encodeWavBlob: function (samples, sampleRate) {
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
        if (audioRecorder.isStarting) {
          reject(new Error("Recording is still starting"));
          return;
        }

        if (!audioRecorder.isRecording) {
          reject(new Error("No active recording"));
          return;
        }

        audioRecorder.isRecording = false;
        let settled = false;

        const finalize = () => {
          if (settled) return;
          settled = true;

          try {
            const samples = audioRecorder.mergePcmChunks();
            const rate = audioRecorder.sampleRate;
            const clampedSamples = audioRecorder.clampSamplesToMax(samples, rate);

            // Always prefer WAV from PCM for the native <audio> player.
            // MediaRecorder webm chunks are often incomplete on stop and grey out the player.
            let playbackBlob = null;
            let mimeType = "audio/wav";
            if (clampedSamples.length > 0) {
              playbackBlob = audioRecorder.encodeWavBlob(clampedSamples, rate);
              mimeType = "audio/wav";
            } else if (audioRecorder.audioBlobs.length > 0) {
              mimeType =
                (audioRecorder.mediaRecorder && audioRecorder.mediaRecorder.mimeType) ||
                audioRecorder.recorderMimeType ||
                "audio/webm";
              playbackBlob = new Blob(audioRecorder.audioBlobs, { type: mimeType });
            }

            audioRecorder.cleanupGraph();
            audioRecorder.stopAllOwnedStreams();
            audioRecorder.pcmChunks = [];
            audioRecorder.resetRecordingProperties();

            if (!playbackBlob || playbackBlob.size === 0) {
              reject(new Error("No audio was captured"));
              return;
            }

            audioRecorder.audioBlobs = [playbackBlob];
            resolve({
              blob: playbackBlob,
              samples: clampedSamples,
              sampleRate: rate,
              mimeType: mimeType,
            });
          } catch (err) {
            audioRecorder.cleanupGraph();
            audioRecorder.stopAllOwnedStreams();
            audioRecorder.resetRecordingProperties();
            reject(err);
          }
        };

        // PCM is the source of truth — no need to wait on MediaRecorder.
        const recorder = audioRecorder.mediaRecorder;
        if (recorder && recorder.state !== "inactive") {
          try { recorder.stop(); } catch (_err) { /* ignore */ }
        }
        setTimeout(finalize, 150);
      });
    },

    cancel: function () {
      audioRecorder.discardChunks = true;
      audioRecorder.isRecording = false;
      audioRecorder.isStarting = false;
      audioRecorder.audioBlobs = [];
      audioRecorder.pcmChunks = [];

      const recorder = audioRecorder.mediaRecorder;
      if (recorder && recorder.state !== "inactive") {
        try { recorder.stop(); } catch (_err) { /* ignore */ }
      }

      audioRecorder.cleanupGraph();
      audioRecorder.stopAllOwnedStreams();
      audioRecorder.resetRecordingProperties();
    },

    stopAllOwnedStreams: function () {
      audioRecorder.ownedStreams.forEach((stream) => {
        try {
          stream.getTracks().forEach((track) => track.stop());
        } catch (_err) {
          /* ignore */
        }
      });
      audioRecorder.ownedStreams.clear();
    },

    stopStream: function () {
      if (!audioRecorder.streamBeingCaptured) return;
      audioRecorder.streamBeingCaptured.getTracks().forEach((track) => track.stop());
    },

    resetRecordingProperties: function () {
      audioRecorder.streamBeingCaptured = null;
      audioRecorder.mediaRecorder = null;
      audioRecorder.isStarting = false;
    },
  };

  return audioRecorder;
}
