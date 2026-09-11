#pragma once
#include <stdint.h>
#include <stddef.h>
#include <string.h>

// ===========================================================================
// wav.h - minimal 16-bit mono PCM WAV writer
//
// Note on endianness: WAV stores PCM little-endian, and the ESP32 is
// little-endian, so an int16_t array can be handed to the encoder as raw
// bytes with no conversion. That is what lets us stream it.
// ===========================================================================

#define WAV_HEADER_BYTES 44

static inline void wavPut32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)(v);
  p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16);
  p[3] = (uint8_t)(v >> 24);
}

static inline void wavPut16(uint8_t *p, uint16_t v) {
  p[0] = (uint8_t)(v);
  p[1] = (uint8_t)(v >> 8);
}

// Total bytes of a complete WAV file holding 'samples' mono int16 samples.
static inline size_t wavByteLength(size_t samples) {
  return WAV_HEADER_BYTES + samples * sizeof(int16_t);
}

// Fills a 44-byte header. Caller supplies the buffer.
static inline void wavBuildHeader(uint8_t *h, size_t samples, uint32_t sampleRate) {
  const uint16_t channels      = 1;
  const uint16_t bitsPerSample = 16;
  const uint32_t dataBytes     = (uint32_t)(samples * 2);
  const uint32_t byteRate      = sampleRate * channels * (bitsPerSample / 8);
  const uint16_t blockAlign    = channels * (bitsPerSample / 8);

  memcpy(h + 0, "RIFF", 4);
  wavPut32(h + 4, 36 + dataBytes);   // size of everything after this field
  memcpy(h + 8, "WAVE", 4);

  memcpy(h + 12, "fmt ", 4);
  wavPut32(h + 16, 16);              // fmt chunk size for PCM
  wavPut16(h + 20, 1);               // format 1 = PCM
  wavPut16(h + 22, channels);
  wavPut32(h + 24, sampleRate);
  wavPut32(h + 28, byteRate);
  wavPut16(h + 32, blockAlign);
  wavPut16(h + 34, bitsPerSample);

  memcpy(h + 36, "data", 4);
  wavPut32(h + 40, dataBytes);
}

// Writes a full WAV (header + PCM) into 'out', which must be at least
// wavByteLength(samples) bytes. Kept for convenience / offline use; the
// streaming publisher in main.ino does not call this, because materialising
// the whole WAV is exactly the copy we are trying to avoid.
static inline void wavWrite(uint8_t *out, const int16_t *pcm,
                            size_t samples, uint32_t sampleRate) {
  wavBuildHeader(out, samples, sampleRate);
  memcpy(out + WAV_HEADER_BYTES, pcm, samples * sizeof(int16_t));
}
