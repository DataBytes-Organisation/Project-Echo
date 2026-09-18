#pragma once

#include "secrets.h"   // WiFi, MQTT, SENSOR_ID, FIXED_LAT/LON

// ===========================================================================
// config.h - everything that isn't a credential
// ===========================================================================

// ---------------- MQTT transport ----------------
// How long esp_mqtt_client waits on a socket operation. A 3 s clip is
// ~128 KB; at the ~68 kB/s measured on your link that's ~2 s of sending,
// so this leaves generous headroom for a weak signal.
#define MQTT_NETWORK_TIMEOUT_MS  30000

// ---------------- Mic pins (validated on your board) ----------------
#define I2S_WS_PIN   15
#define I2S_SCK_PIN  16
#define I2S_SD_PIN   17
#define I2S_PORT     I2S_NUM_0

// We capture BOTH I2S slots and keep one. The diagnostic run showed live
// audio on slot1 with L/R strapped to GND. Reading both and picking the
// live one sidesteps the legacy driver's left/right polarity quirk.
//   slot0 = WS-low half-frame, slot1 = WS-high half-frame
#define MIC_SLOT     1

// ---------------- Audio rates ----------------
// SAMPLE_RATE is what goes in the WAV header and what the Engine expects.
#define SAMPLE_RATE      16000

// The mic runs at SAMPLE_RATE * MIC_DECIMATION and we average groups of
// MIC_DECIMATION samples down to SAMPLE_RATE. 48 kHz is the INMP441's
// nominal rate and divides by 3 exactly, so there's no resampling error.
// Must divide MIC_FRAMES_PER_READ (384): so 1, 2, 3, 4, 6 or 8.
#define MIC_DECIMATION   3
#define MIC_SAMPLE_RATE  (SAMPLE_RATE * MIC_DECIMATION)   // 48000

// ---------------- Clip length and timing ----------------
// 3 s at 16 kHz = 96,000 samples = ~128 KB of JSON once base64'd.
// Buffers cost roughly 75 KB per second of clip, so 3 s uses ~225 KB
// of the 2 MB PSRAM.
#define RECORD_SECONDS   3
#define RECORD_SAMPLES   (SAMPLE_RATE * RECORD_SECONDS)

// Start a new recording every CYCLE_PERIOD_MS. This is a PERIOD, not a gap:
// the wait shrinks to absorb however long recording and publishing took, so
// clips start on a steady 5 s beat.
//
// Budget check: 3 s record + ~2 s publish = ~5 s, so a clip that PASSES the
// gate leaves almost no slack and may overrun slightly. Clips that fail the
// gate skip the publish entirely and finish in ~3 s. Watch for [loop]
// overran lines - if you see them constantly, raise this to 6000-7000.
#define CYCLE_PERIOD_MS  5000

// ---------------- Gain and DC ----------------
// The INMP441 is quiet. Gain 12 put your test recordings around -19 dBFS
// RMS, which is healthy, but peaks were touching the rail (0.3% clipped).
// If clipping keeps showing up in the [mic] line, drop this to 8.
#define MIC_GAIN         8

// One-pole DC blocker. The INMP441 has a real DC offset that would
// otherwise eat headroom and add a click at the start of every clip.
#define MIC_DC_BLOCK     1
#define MIC_DC_R         0.995f

// ---------------- Publish gate ----------------
// A clip is only published if it passes ALL three. Everything else is
// discarded on the device and never costs airtime.
//
//   RMS   - overall loudness. Below this is silence or room tone.
//   CLIP% - samples pinned at the rail. Usually wind hitting the capsule.
//   HFR   - rms(sample-to-sample difference) / rms(signal). A cheap
//           spectral centroid: for a tone at f it equals 2*sin(pi*f/fs).
//           0.45 is roughly a 1.2 kHz centroid. Below that the energy is
//           all low-frequency (steady wind, rumble, hum) and not bird-shaped.
//
// These are starting points from synthetic test signals. Watch the [gate]
// lines against clips you can actually hear and retune.
#define GATE_MIN_RMS       150.0
#define GATE_MAX_CLIP_PCT  1.0
#define GATE_MIN_HFR       0.00

// 1 = compute and log the gate verdict but publish EVERYTHING anyway.
//     Run this for a session before trusting the thresholds: compare the
//     [gate] verdicts against what you can hear in clips/, then set it
//     back to 0. A clip you never send is gone for good.
#define GATE_LOG_ONLY      0

// ---------------- Test switches ----------------
// 1 = record and print stats, but never touch WiFi/MQTT.
#define SKIP_MQTT              0

// 1 = dump the base64 payload to serial between markers, so you can paste
//     it into `python test_publisher.py decode` and get a WAV without MQTT
//     involved at all. Slow (~128 KB of serial at 3 s).
#define DUMP_BASE64_TO_SERIAL  0

// 1 = sync the clock over NTP so timestamps are real epochs rather than
//     seconds-since-boot.
#define USE_NTP                1