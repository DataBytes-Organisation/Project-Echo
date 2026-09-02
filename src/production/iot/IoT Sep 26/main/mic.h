#pragma once
#include <Arduino.h>
#include <math.h>
#include "driver/i2s.h"
#include "config.h"

// ===========================================================================
// mic.h - INMP441 capture on ESP32-S3
//
// Captures 32-bit stereo frames, keeps MIC_SLOT, sign-extends the 24-bit
// sample, removes DC, applies gain, clamps to int16.
// ===========================================================================

// 384 frames per read = 768 int32 words (stereo) = 3072 bytes.
// 384 is divisible by 1, 2, 3, 4, 6 and 8, so decimation never straddles a
// block boundary and we don't need carry-over state between reads.
#define MIC_FRAMES_PER_READ 384
#define MIC_WORDS_PER_READ  (MIC_FRAMES_PER_READ * 2)

#define MIC_DMA_BUF_COUNT 8
#define MIC_DMA_BUF_LEN   384

struct MicStats {
  size_t   samples;
  int16_t  minV;
  int16_t  maxV;
  int32_t  peak;        // max |sample|
  double   rms;
  double   rmsDiff;     // RMS of the first difference
  double   hfr;         // rmsDiff / rms - cheap spectral centroid proxy
  uint32_t clipped;     // samples that hit the int16 rail
  uint32_t zeros;
  uint32_t elapsedMs;
  double   effectiveRate;
};

static MicStats g_micStats;
static bool     g_micReady = false;

static int32_t  g_micRaw[MIC_WORDS_PER_READ];

// DC blocker state
static float g_dcX1 = 0.0f;
static float g_dcY1 = 0.0f;

static inline const MicStats& micLastStats() { return g_micStats; }

// ---------------------------------------------------------------------------

static bool micBegin() {
  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = MIC_SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,   // 24-bit data in 32-bit frame
    .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,   // both slots, we pick one
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = MIC_DMA_BUF_COUNT,
    .dma_buf_len = MIC_DMA_BUF_LEN,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  esp_err_t err = i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
  if (err != ESP_OK) {
    Serial.printf("[mic] i2s_driver_install failed: %d\n", (int)err);
    return false;
  }

  i2s_pin_config_t pins = {
    .bck_io_num   = I2S_SCK_PIN,
    .ws_io_num    = I2S_WS_PIN,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num  = I2S_SD_PIN
  };

  err = i2s_set_pin(I2S_PORT, &pins);
  if (err != ESP_OK) {
    Serial.printf("[mic] i2s_set_pin failed: %d\n", (int)err);
    i2s_driver_uninstall(I2S_PORT);
    return false;
  }

  i2s_zero_dma_buffer(I2S_PORT);
  g_micReady = true;

  Serial.printf("[mic] ready: %d Hz capture, slot%d, decimate %d -> %d Hz out, gain %d\n",
                MIC_SAMPLE_RATE, MIC_SLOT, MIC_DECIMATION, SAMPLE_RATE, MIC_GAIN);
  return true;
}

// ---------------------------------------------------------------------------

// Throw away the first `ms` of audio. The mic needs roughly 50 ms of bit
// clock before its output settles, and the DMA ring may still hold zeros
// from i2s_zero_dma_buffer().
static void micWarmUp(uint32_t ms) {
  uint32_t until = millis() + ms;
  size_t br;
  while (millis() < until) {
    if (i2s_read(I2S_PORT, (void *)g_micRaw, sizeof(g_micRaw),
                 &br, pdMS_TO_TICKS(200)) != ESP_OK) break;
  }
}

// Records up to maxSamples int16 mono samples into `out`.
// Returns the number actually captured.
static size_t micRecord(int16_t *out, size_t maxSamples) {
  if (!g_micReady) {
    Serial.println("[mic] micRecord called before micBegin succeeded");
    return 0;
  }

  micWarmUp(60);

  g_dcX1 = 0.0f;
  g_dcY1 = 0.0f;
  bool primed = false;

  size_t   produced = 0;
  int64_t  sumSq    = 0;
  int64_t  sumSqDiff = 0;
  int16_t  prev     = 0;
  bool     havePrev = false;
  int32_t  peak     = 0;
  int16_t  minV     = 32767;
  int16_t  maxV     = -32768;
  uint32_t clipped  = 0;
  uint32_t zeros    = 0;

  uint32_t t0 = millis();

  while (produced < maxSamples) {
    size_t bytesRead = 0;
    esp_err_t err = i2s_read(I2S_PORT, (void *)g_micRaw, sizeof(g_micRaw),
                             &bytesRead, pdMS_TO_TICKS(1000));
    if (err != ESP_OK || bytesRead == 0) {
      Serial.println("[mic] i2s_read timed out mid-recording");
      break;
    }

    size_t frames = (bytesRead / sizeof(int32_t)) / 2;

    for (size_t f = 0; f + MIC_DECIMATION <= frames && produced < maxSamples;
         f += MIC_DECIMATION) {

      // Average MIC_DECIMATION consecutive frames. With MIC_DECIMATION == 1
      // this is just a plain read, and the compiler folds the loop away.
      int32_t acc = 0;
      for (int k = 0; k < MIC_DECIMATION; k++) {
        // Arithmetic shift on a signed type preserves the sign bit.
        acc += (g_micRaw[(f + k) * 2 + MIC_SLOT] >> 8);
      }
      float x = (float)acc / (float)MIC_DECIMATION;   // 24-bit domain

#if MIC_DC_BLOCK
      // Prime the filter state from the first sample. Starting from zero
      // pushes the mic's entire DC offset through the first output sample
      // at full amplitude, producing a decaying thump ~200 samples long
      // that dominates the peak reading and clicks on playback.
      if (!primed) {
        g_dcX1 = x;
        g_dcY1 = 0.0f;
        primed = true;
      }
      float y = x - g_dcX1 + MIC_DC_R * g_dcY1;
      g_dcX1 = x;
      g_dcY1 = y;
#else
      float y = x;
#endif

      // 24-bit -> 16-bit is a divide by 256, then apply gain.
      int32_t v = (int32_t)lrintf((y / 256.0f) * (float)MIC_GAIN);
      if (v > 32767)  { v = 32767;  clipped++; }
      if (v < -32768) { v = -32768; clipped++; }

      int16_t s = (int16_t)v;
      out[produced++] = s;

      // Differencing is a +6 dB/octave high-pass. The ratio of its RMS to
      // the signal RMS tracks the dominant frequency without an FFT:
      // for a sine at f it equals exactly 2*sin(pi*f/fs).
      if (havePrev) {
        int32_t d = (int32_t)s - (int32_t)prev;
        sumSqDiff += (int64_t)d * (int64_t)d;
      }
      prev = s;
      havePrev = true;

      if (s == 0) zeros++;
      if (s < minV) minV = s;
      if (s > maxV) maxV = s;
      int32_t a = (s < 0) ? -(int32_t)s : (int32_t)s;
      if (a > peak) peak = a;
      sumSq += (int64_t)s * (int64_t)s;
    }
  }

  uint32_t elapsed = millis() - t0;

  g_micStats.samples       = produced;
  g_micStats.minV          = (produced ? minV : 0);
  g_micStats.maxV          = (produced ? maxV : 0);
  g_micStats.peak          = peak;
  g_micStats.rms           = produced ? sqrt((double)sumSq / (double)produced) : 0.0;
  g_micStats.rmsDiff       = produced ? sqrt((double)sumSqDiff / (double)produced) : 0.0;
  g_micStats.hfr           = (g_micStats.rms > 0.0)
                               ? g_micStats.rmsDiff / g_micStats.rms : 0.0;
  g_micStats.clipped       = clipped;
  g_micStats.zeros         = zeros;
  g_micStats.elapsedMs     = elapsed;
  g_micStats.effectiveRate = elapsed ? (1000.0 * (double)produced / (double)elapsed) : 0.0;

  return produced;
}

// ---------------------------------------------------------------------------

// Prints the stats and, more usefully, a verdict on whether what we captured
// is plausibly real audio rather than silence, a stuck line, or clipping.
static void micPrintStats(const int16_t *pcm) {
  const MicStats &s = g_micStats;

  Serial.printf("[mic] %u samples in %lu ms (effective %.0f Hz, expected %d Hz)\n",
                (unsigned)s.samples, (unsigned long)s.elapsedMs,
                s.effectiveRate, SAMPLE_RATE);
  Serial.printf("[mic] min=%d max=%d peak=%ld rms=%.1f hfr=%.2f clipped=%u zeros=%u\n",
                s.minV, s.maxV, (long)s.peak, s.rms, s.hfr,
                (unsigned)s.clipped, (unsigned)s.zeros);

  if (s.samples >= 8) {
    Serial.print("[mic] first 8 PCM: ");
    for (int i = 0; i < 8; i++) Serial.printf("%d ", pcm[i]);
    Serial.println();
  }

  double dbfs = (s.peak > 0) ? 20.0 * log10((double)s.peak / 32768.0) : -999.0;
  Serial.printf("[mic] peak level: %.1f dBFS\n", dbfs);

  // ---- verdicts ----
  if (s.samples == 0) {
    Serial.println("[mic] FAIL: captured nothing. I2S read is failing.");
  } else if (s.peak == 0) {
    Serial.println("[mic] FAIL: pure digital silence. Check MIC_SLOT (try 0), "
                   "and that the mic still has power.");
  } else if (s.rms < 5.0) {
    Serial.println("[mic] WARN: essentially silent. Either the room is very "
                   "quiet, MIC_GAIN is too low, or MIC_SLOT is wrong.");
  } else if (s.clipped > s.samples / 100) {
    Serial.printf("[mic] WARN: %.1f%% of samples clipped. Lower MIC_GAIN.\n",
                  100.0 * (double)s.clipped / (double)s.samples);
  } else {
    Serial.println("[mic] OK: looks like real audio.");
  }

  // A big mismatch here means the DMA is underrunning or the mic is not
  // actually clocking at the rate we asked for, which produces audio that
  // plays back at the wrong speed.
  if (s.effectiveRate > 0 &&
      fabs(s.effectiveRate - (double)SAMPLE_RATE) > (double)SAMPLE_RATE * 0.05) {
    Serial.println("[mic] WARN: effective rate is >5% off the configured rate. "
                   "Playback will sound pitch-shifted.");
  }
}
