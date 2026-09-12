/**
 * FR-B2 duration-cap tests for audio_recorder helpers.
 * Run: node --test public/js/audio_recorder.test.mjs
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  MAX_RECORDING_SECONDS,
  clampSamplesToMax,
  encodeWavBlob,
  buildCappedRecording,
} from "./audio_recorder.js";

function makeSamples(seconds, sampleRate) {
  return new Float32Array(Math.floor(seconds * sampleRate));
}

function durationSeconds(samples, sampleRate) {
  return samples.length / sampleRate;
}

describe("FR-B2 recording duration cap", () => {
  const sampleRate = 44100;

  it("clampSamplesToMax keeps recordings at or under 20 seconds", () => {
    const longSamples = makeSamples(35, sampleRate);
    const capped = clampSamplesToMax(longSamples, sampleRate);

    assert.ok(durationSeconds(capped, sampleRate) <= MAX_RECORDING_SECONDS);
    assert.equal(capped.length, Math.floor(sampleRate * MAX_RECORDING_SECONDS));
  });

  it("PCM path returns a WAV blob of 20 seconds or less", async () => {
    const pcmSamples = makeSamples(28, sampleRate);
    const result = await buildCappedRecording({
      pcmSamples,
      sampleRate,
      mediaBlobs: [new Blob([new Uint8Array([1, 2, 3])], { type: "audio/webm" })],
      mediaMimeType: "audio/webm",
      decodeAudioBlob: async () => {
        throw new Error("decode should not run when PCM is present");
      },
    });

    assert.equal(result.mimeType, "audio/wav");
    assert.ok(result.blob instanceof Blob);
    assert.equal(result.blob.type, "audio/wav");
    assert.ok(durationSeconds(result.samples, result.sampleRate) <= MAX_RECORDING_SECONDS);
    assert.equal(
      result.samples.length,
      Math.floor(sampleRate * MAX_RECORDING_SECONDS)
    );
  });

  it("no-PCM MediaRecorder fallback clamps decoded samples to 20 seconds", async () => {
    const decodedRate = 48000;
    const decodedSamples = makeSamples(40, decodedRate);

    const result = await buildCappedRecording({
      pcmSamples: new Float32Array(0),
      sampleRate,
      mediaBlobs: [new Blob([new Uint8Array([9, 9, 9])], { type: "audio/webm" })],
      mediaMimeType: "audio/webm",
      decodeAudioBlob: async (_blob) => ({
        samples: decodedSamples,
        sampleRate: decodedRate,
      }),
    });

    assert.equal(result.mimeType, "audio/wav");
    assert.equal(result.sampleRate, decodedRate);
    assert.ok(durationSeconds(result.samples, result.sampleRate) <= MAX_RECORDING_SECONDS);
    assert.equal(
      result.samples.length,
      Math.floor(decodedRate * MAX_RECORDING_SECONDS)
    );

    // Encoded WAV should match the capped sample length (44-byte header + 2 bytes/sample).
    const wavBytes = await result.blob.arrayBuffer();
    assert.equal(wavBytes.byteLength, 44 + result.samples.length * 2);
  });

  it("fallback decode failure returns a friendly error, not an uncapped blob", async () => {
    await assert.rejects(
      () =>
        buildCappedRecording({
          pcmSamples: new Float32Array(0),
          sampleRate,
          mediaBlobs: [new Blob([new Uint8Array([1])], { type: "audio/webm" })],
          mediaMimeType: "audio/webm",
          decodeAudioBlob: async () => {
            throw new Error("decodeAudioData failed");
          },
        }),
      (err) => {
        assert.match(err.message, /Could not process the recorded audio/i);
        return true;
      }
    );
  });

  it("encodeWavBlob produces a non-empty audio/wav blob", () => {
    const samples = makeSamples(1, sampleRate);
    const blob = encodeWavBlob(samples, sampleRate);
    assert.equal(blob.type, "audio/wav");
    assert.ok(blob.size > 44);
  });
});
