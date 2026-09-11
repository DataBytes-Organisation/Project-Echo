#include <WiFi.h>
#include <time.h>
#include <sys/time.h>           // gettimeofday() - same clock as time.h, but
                                // with the microseconds the new format needs
#include "mqtt_client.h"        // ESP-IDF MQTT - bundled with the ESP32 core
#include "esp_idf_version.h"

#include "config.h"
#include "mic.h"
#include "wav.h"
#include "power_placeholders.h"

// ===========================================================================
// ESP32-S3 -> MQTT audio node
//
// Records a RECORD_SECONDS clip every CYCLE_PERIOD_MS, and publishes it
// only if it passes the gate (loud enough, not saturated, not all
// low-frequency rumble). Everything else is discarded on the device.
//
// Payload shape is fixed by the Engine team:
//   { "timestamp": "2026-08-01T09:58:09.523724", "sensorId": "...",
//     "microphoneLLA": [lat, lon, alt], "animalEstLLA": [...],
//     "animalTrueLLA": [...], "animalLLAUncertainty": 0,
//     "audioClip": "<base64 wav>", "audioFile": "Animal_Mode" }
//
// Watch the last two: audioFile is a mode LABEL, the audio goes in audioClip.
// The names read as interchangeable and are not.
//
// Design: the whole JSON payload is built in one flat PSRAM buffer and
// published in a single call. esp_mqtt_client has no 16-bit length limit -
// PubSubClient capped every message at 65,535 bytes, which is what used to
// restrict clips to 1.5 s.
//
// Memory (3 s clip @ 16 kHz):
//   wavBuffer    96,044 B   [44-byte header][PCM] - mic records straight in
//   jsonBuffer  128,636 B   prefix + base64(wavBuffer) + suffix
//   total       ~225 KB of the 2 MB PSRAM. Internal SRAM stays free for WiFi.
// ===========================================================================

static uint8_t *wavBuffer  = nullptr;  // header + PCM, contiguous
static int16_t *pcmBuffer  = nullptr;  // points INTO wavBuffer at offset 44
static char    *jsonBuffer = nullptr;  // the finished payload

// Everything in the payload except the audio. The prefix runs ~240 B with
// realistic coordinates; 512 leaves room for a longer SENSOR_ID or wider
// values. Both the snprintf cap in captureAndPublish() and the buffer size in
// setup() read these, so the guard can't disagree with the allocation - which
// is the failure that would overrun the buffer silently.
#define JSON_PREFIX_MAX  512
#define JSON_SUFFIX      "\",\"audioFile\":\"Animal_Mode\"}"

static esp_mqtt_client_handle_t mqttClient = nullptr;
static volatile bool mqttUp = false;

// Running tally, printed each cycle so you can see the gate's hit rate.
static uint32_t nRecorded = 0;
static uint32_t nPublished = 0;

// ---------------------------------------------------------------------------
// Base64
// ---------------------------------------------------------------------------

static const char B64_ALPHABET[] =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// Length of the base64 encoding of n bytes, including '=' padding.
static inline size_t b64Length(size_t n) {
  return 4 * ((n + 2) / 3);
}

// Encodes in[0..n) into out. Returns characters written (== b64Length(n)).
static size_t b64Encode(const uint8_t *in, size_t n, char *out) {
  size_t o = 0, i = 0;

  for (; i + 3 <= n; i += 3) {
    uint32_t v = ((uint32_t)in[i] << 16) | ((uint32_t)in[i + 1] << 8) | in[i + 2];
    out[o++] = B64_ALPHABET[(v >> 18) & 63];
    out[o++] = B64_ALPHABET[(v >> 12) & 63];
    out[o++] = B64_ALPHABET[(v >> 6) & 63];
    out[o++] = B64_ALPHABET[v & 63];
  }

  size_t rem = n - i;
  if (rem == 1) {
    uint32_t v = (uint32_t)in[i] << 16;
    out[o++] = B64_ALPHABET[(v >> 18) & 63];
    out[o++] = B64_ALPHABET[(v >> 12) & 63];
    out[o++] = '=';
    out[o++] = '=';
  } else if (rem == 2) {
    uint32_t v = ((uint32_t)in[i] << 16) | ((uint32_t)in[i + 1] << 8);
    out[o++] = B64_ALPHABET[(v >> 18) & 63];
    out[o++] = B64_ALPHABET[(v >> 12) & 63];
    out[o++] = B64_ALPHABET[(v >> 6) & 63];
    out[o++] = '=';
  }
  return o;
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

// Big buffers go to PSRAM so internal SRAM stays free for the WiFi stack.
static void *bigAlloc(size_t n, const char *what) {
  if (psramFound()) {
    void *p = ps_malloc(n);
    if (p) {
      Serial.printf("[mem] %s: %u B in PSRAM (%u B PSRAM left)\n",
                    what, (unsigned)n, (unsigned)ESP.getFreePsram());
      return p;
    }
    Serial.printf("[mem] %s: ps_malloc(%u) failed, only %u B PSRAM free\n",
                  what, (unsigned)n, (unsigned)ESP.getFreePsram());
    return nullptr;
  }

  // No PSRAM. Only try internal SRAM if it could plausibly fit - a doomed
  // malloc just produces a confusing second failure.
  size_t largest = ESP.getMaxAllocHeap();
  if (n > largest) {
    Serial.printf("[mem] %s: need %u B but largest free internal block is %u B\n",
                  what, (unsigned)n, (unsigned)largest);
    return nullptr;
  }
  void *p = malloc(n);
  Serial.printf("[mem] %s: %u B in internal SRAM%s\n",
                what, (unsigned)n, p ? "" : " - FAILED");
  return p;
}

// ---------------------------------------------------------------------------
// Network
// ---------------------------------------------------------------------------

void connectWiFi() {
  Serial.printf("[wifi] connecting to %s", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    if (millis() - start > 30000) {
      Serial.println("\n[wifi] still not connected after 30s, retrying");
      WiFi.disconnect();
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
      start = millis();
    }
  }
  Serial.printf("\n[wifi] connected, IP=%s RSSI=%d\n",
                WiFi.localIP().toString().c_str(), WiFi.RSSI());
}

void syncTime() {
#if USE_NTP
  configTime(0, 0, "pool.ntp.org", "time.nist.gov");
  Serial.print("[time] syncing");
  uint32_t start = millis();
  while (time(nullptr) < 1700000000UL && millis() - start < 8000) {
    delay(250);
    Serial.print(".");
  }
  time_t now = time(nullptr);
  if (now > 1700000000UL) {
    Serial.printf("\n[time] epoch=%lu\n", (unsigned long)now);
  } else {
    Serial.println("\n[time] NTP failed, falling back to millis()");
  }
#endif
}

// ISO 8601 with microseconds, UTC: "2026-08-01T09:58:09.523724".
//
// If syncTime() succeeded this is real wall-clock time. If it didn't,
// gettimeofday() still answers - it just counts from the epoch at power-on, so
// you get a date in 1970. That's left visible on purpose: an obviously broken
// timestamp beats a plausible wrong one that nobody notices for weeks.
//
// The sample payload has no timezone suffix, so UTC is the assumption. Confirm
// with the Engine team before deployment; a 10 h AEST offset is easy to miss.
static void currentTimestampIso(char *out, size_t n) {
  struct timeval tv;
  gettimeofday(&tv, nullptr);

  struct tm tmv;
  gmtime_r(&tv.tv_sec, &tmv);

  // strftime has no microseconds field, so the fraction is appended by hand.
  // %06ld pads to six digits - the Engine expects a fixed six, and .42 would
  // not match.
  size_t k = strftime(out, n, "%Y-%m-%dT%H:%M:%S", &tmv);
  snprintf(out + k, n - k, ".%06ld", (long)tv.tv_usec);
}

// ---------------------------------------------------------------------------
// MQTT (esp_mqtt_client runs in its own task and reconnects by itself)
// ---------------------------------------------------------------------------

static void mqttEvent(void *args, esp_event_base_t base,
                      int32_t event_id, void *event_data) {
  switch ((esp_mqtt_event_id_t)event_id) {
    case MQTT_EVENT_CONNECTED:
      mqttUp = true;
      Serial.println("[mqtt] connected");
      break;
    case MQTT_EVENT_DISCONNECTED:
      mqttUp = false;
      Serial.println("[mqtt] disconnected (auto-reconnect is on)");
      break;
    case MQTT_EVENT_ERROR:
      Serial.println("[mqtt] transport error");
      break;
    default:
      break;
  }
}

void startMQTT() {
  static char uri[96];
  static char clientId[48];
  snprintf(uri, sizeof(uri), "mqtt://%s:%d", MQTT_BROKER, MQTT_PORT);
  snprintf(clientId, sizeof(clientId), "%s-%08lx",
           SENSOR_ID, (unsigned long)esp_random());

  esp_mqtt_client_config_t cfg = {};
  // The config struct was reorganised between ESP-IDF 4 (Arduino core 2.x)
  // and ESP-IDF 5 (Arduino core 3.x). Support both.
#if ESP_IDF_VERSION_MAJOR >= 5
  cfg.broker.address.uri = uri;
  cfg.credentials.client_id = clientId;
  cfg.network.timeout_ms = MQTT_NETWORK_TIMEOUT_MS;
#else
  cfg.uri = uri;
  cfg.client_id = clientId;
  cfg.network_timeout_ms = MQTT_NETWORK_TIMEOUT_MS;
#endif

  mqttClient = esp_mqtt_client_init(&cfg);
  if (!mqttClient) {
    Serial.println("[mqtt] FATAL: client init failed");
    return;
  }
  esp_mqtt_client_register_event(mqttClient,
                                 (esp_mqtt_event_id_t)ESP_EVENT_ANY_ID,
                                 mqttEvent, NULL);
  esp_mqtt_client_start(mqttClient);

  Serial.printf("[mqtt] connecting to %s\n", uri);
  uint32_t t0 = millis();
  while (!mqttUp && millis() - t0 < 15000) delay(100);
  if (!mqttUp) {
    Serial.println("[mqtt] not up yet - it keeps retrying in the background, "
                   "publishes are skipped until it connects");
  }
}

// ---------------------------------------------------------------------------
// One cycle: record -> gate -> maybe publish
// ---------------------------------------------------------------------------

void captureAndPublish() {
  // ---- 1. record (straight into wavBuffer, after the header space) ----
  Serial.printf("\n[mic] recording %d s...\n", RECORD_SECONDS);
  size_t samples = micRecord(pcmBuffer, RECORD_SAMPLES);
  if (samples == 0) {
    Serial.println("[err] no samples captured, skipping");
    return;
  }
  nRecorded++;
  micPrintStats(pcmBuffer);

  // ---- 2. gate ----
  const MicStats &st = micLastStats();
  double clipPct = st.samples ? 100.0 * (double)st.clipped / (double)st.samples : 0.0;

  const char *why = nullptr;
  if      (st.rms < GATE_MIN_RMS)        why = "too quiet";
  else if (clipPct > GATE_MAX_CLIP_PCT)  why = "saturated";
  else if (st.hfr < GATE_MIN_HFR)        why = "low-freq (wind/rumble)";

  if (why) {
    Serial.printf("[gate] %s (rms=%.0f clip=%.2f%% hfr=%.2f) - %s\n",
                  GATE_LOG_ONLY ? "would SKIP" : "SKIP",
                  st.rms, clipPct, st.hfr, why);
#if !GATE_LOG_ONLY
    Serial.printf("[stat] %lu recorded, %lu published (%.0f%% kept)\n",
                  (unsigned long)nRecorded, (unsigned long)nPublished,
                  100.0 * nPublished / nRecorded);
    return;                       // <-- discarded here, costs no airtime
#endif
  } else {
    Serial.printf("[gate] KEEP (rms=%.0f clip=%.2f%% hfr=%.2f)\n",
                  st.rms, clipPct, st.hfr);
  }

  // ---- 3. build the whole payload in jsonBuffer ----
  size_t wavLen = wavByteLength(samples);
  wavBuildHeader(wavBuffer, samples, SAMPLE_RATE);   // PCM is already in place

  // Altitude belongs in secrets.h beside FIXED_LAT/FIXED_LON. The fallback
  // keeps the sketch building against an older secrets.h; survey the real
  // value before deployment, since 0 reads as sea level.
#ifndef FIXED_ALT
#define FIXED_ALT 0.0f
#endif

  // The spec order puts everything except a short tail before the audio, so
  // the payload is still [prefix][base64][suffix] in one buffer, with
  // b64Encode writing straight into its final position.
  char ts[40];
  currentTimestampIso(ts, sizeof ts);

  char *p = jsonBuffer;
  int pre = snprintf(p, JSON_PREFIX_MAX,
                     "{\"timestamp\":\"%s\","
                     "\"sensorId\":\"%s\","
                     "\"microphoneLLA\":[%.7f,%.7f,%.2f],"
                     "\"animalEstLLA\":[%.7f,%.7f,%.2f],"
                     "\"animalTrueLLA\":[%.7f,%.7f,%.2f],"
                     "\"animalLLAUncertainty\":%.1f,"
                     "\"audioClip\":\"",
                     ts,
                     SENSOR_ID,
                     (double)FIXED_LAT, (double)FIXED_LON, (double)FIXED_ALT,
                     // Nothing on this device locates the animal - that needs
                     // several mics or the Engine's inference. Placeholders so
                     // the payload has the required shape. animalTrueLLA is
                     // ground truth and only exists in controlled tests.
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     // Metres. 0 here means "no estimate", not "perfect
                     // estimate" - worth agreeing on with the Engine team,
                     // since the two read identically on the wire.
                     0.0);

  // snprintf returns what it WOULD have written, so >= the limit means it was
  // cut short: an unclosed JSON string the far end can't parse. Drop the clip
  // rather than publish 128 KB that fails silently at the Engine.
  if (pre < 0 || pre >= (int)JSON_PREFIX_MAX) {
    Serial.printf("[err] JSON prefix needed %d B, cap is %u - skipping clip\n",
                  pre, (unsigned)JSON_PREFIX_MAX);
    return;
  }

  size_t prefixLen = (size_t)pre;   // name kept: the DUMP block below prints
                                    // from jsonBuffer + prefixLen
  p += prefixLen;

  size_t b64Len = b64Encode(wavBuffer, wavLen, p);
  p += b64Len;

  // No JSON escaping needed: base64 output is only A-Z a-z 0-9 + / =,
  // none of which are special inside a JSON string.
  //
  // sizeof counts the literal's terminating zero, hence -1. MQTT publishes an
  // explicit length rather than stopping at a zero, so that byte would just be
  // a stray character inside the JSON.
  memcpy(p, JSON_SUFFIX, sizeof(JSON_SUFFIX) - 1);
  p += sizeof(JSON_SUFFIX) - 1;

  size_t totalLen = (size_t)(p - jsonBuffer);

  Serial.printf("[pay] wav=%u B  base64=%u B  json=%u B\n",
                (unsigned)wavLen, (unsigned)b64Len, (unsigned)totalLen);

#if DUMP_BASE64_TO_SERIAL
  Serial.println("----- BEGIN BASE64 WAV -----");
  Serial.write((const uint8_t *)(jsonBuffer + prefixLen), b64Len);
  Serial.println("\n----- END BASE64 WAV -----");
#endif

#if SKIP_MQTT
  Serial.println("[mqtt] SKIP_MQTT is set, not publishing");
#else
  if (!mqttUp) {
    Serial.println("[mqtt] not connected, dropping this clip");
  } else {
    uint32_t t0 = millis();
    int msgId = esp_mqtt_client_publish(mqttClient, MQTT_TOPIC,
                                        jsonBuffer, totalLen,
                                        /*qos=*/0, /*retain=*/0);
    uint32_t dt = millis() - t0;

    if (msgId >= 0) {
      nPublished++;
      Serial.printf("[mqtt] published %u B in %lu ms (%.1f kB/s)\n",
                    (unsigned)totalLen, (unsigned long)dt,
                    dt ? (totalLen / 1024.0) / (dt / 1000.0) : 0.0);
    } else {
      Serial.println("[mqtt] publish FAILED");
      Serial.println("       If this repeats at this clip length but works");
      Serial.println("       shorter, the broker is rejecting the size -");
      Serial.println("       halve RECORD_SECONDS and retest.");
    }
  }
#endif

  Serial.printf("[stat] %lu recorded, %lu published (%.0f%% kept)\n",
                (unsigned long)nRecorded, (unsigned long)nPublished,
                100.0 * nPublished / nRecorded);
}

// ---------------------------------------------------------------------------

void setup() {
  Serial.begin(115200);

#if ARDUINO_USB_CDC_ON_BOOT
  // The S3's USB CDC driver DISCARDS output when the host isn't draining it
  // fast enough. Make it wait instead. Set to 0 for headless deployment.
  Serial.setTxTimeoutMs(1000);

  // After a reset the host has to re-enumerate the USB device, which takes
  // 1-2 s on Windows. Anything printed before that is lost, which looks
  // exactly like a sketch that never started.
  uint32_t serialWait = millis();
  while (!Serial && millis() - serialWait < 4000) delay(10);
  delay(300);
#else
  delay(800);
#endif

  Serial.println("\n\n=== ESP32-S3 audio node ===");
  Serial.printf("[boot] %d s clip every %d ms, gate %s\n",
                RECORD_SECONDS, CYCLE_PERIOD_MS,
                GATE_LOG_ONLY ? "LOGGING ONLY (publishing everything)" : "active");
  Serial.flush();

  // wavBuffer holds [header][PCM] contiguously; the mic records straight
  // into the PCM part, so the WAV never needs assembling or copying.
  size_t wavBytes  = WAV_HEADER_BYTES + (size_t)RECORD_SAMPLES * sizeof(int16_t);
  size_t jsonBytes = JSON_PREFIX_MAX + b64Length(wavBytes)
                     + sizeof(JSON_SUFFIX);   // sizeof includes the NUL, which
                                              // covers the terminator snprintf
                                              // writes while building the prefix

  if (psramFound()) {
    Serial.printf("[mem] PSRAM: %u B total, %u B free\n",
                  (unsigned)ESP.getPsramSize(), (unsigned)ESP.getFreePsram());
  } else {
    Serial.println("[mem] PSRAM: NOT DETECTED");
  }
  Serial.printf("[mem] internal heap: %u B free, largest block %u B\n",
                (unsigned)ESP.getFreeHeap(), (unsigned)ESP.getMaxAllocHeap());
  Serial.printf("[mem] need: wav %u B + json %u B = %u B\n",
                (unsigned)wavBytes, (unsigned)jsonBytes,
                (unsigned)(wavBytes + jsonBytes));

  wavBuffer  = (uint8_t *)bigAlloc(wavBytes, "wav buffer");
  jsonBuffer = (char *)bigAlloc(jsonBytes, "json buffer");
  if (!wavBuffer || !jsonBuffer) {
    Serial.println();
    if (!psramFound()) {
      Serial.println("[mem] FATAL: PSRAM is not enabled in this build.");
      Serial.println("  Tools > PSRAM > QSPI PSRAM   (2 MB / N8R2 modules)");
      Serial.println("  Tools > PSRAM > OPI PSRAM    (8 MB / N16R8 only)");
    } else {
      Serial.println("[mem] FATAL: PSRAM present but too small for this clip.");
      Serial.printf("  Lower RECORD_SECONDS - it costs about %u B per second.\n",
                    (unsigned)((wavBytes + jsonBytes) / RECORD_SECONDS));
    }
    while (true) delay(1000);
  }
  pcmBuffer = (int16_t *)(wavBuffer + WAV_HEADER_BYTES);

  connectSolar();
  connect4G();

  if (!micBegin()) {
    Serial.println("[mic] FATAL: I2S would not start");
    while (true) delay(1000);
  }

#if !SKIP_MQTT
  connectWiFi();
  syncTime();
  startMQTT();
#else
  Serial.println("[net] SKIP_MQTT set, skipping WiFi and MQTT entirely");
#endif

  Serial.println("\n[boot] setup complete, entering main loop");
}

void loop() {
  // No mqtt.loop() and no manual reconnect: esp_mqtt_client runs in its own
  // FreeRTOS task and handles keepalive and reconnection itself.
  uint32_t cycleStart = millis();

  captureAndPublish();

  // CYCLE_PERIOD_MS is a period, not a gap: the wait absorbs however long
  // recording and publishing took, so clips start on a steady beat.
  uint32_t elapsed = millis() - cycleStart;
  if (elapsed < CYCLE_PERIOD_MS) {
    uint32_t wait = CYCLE_PERIOD_MS - elapsed;
    Serial.printf("[loop] cycle took %lu ms, waiting %lu ms\n",
                  (unsigned long)elapsed, (unsigned long)wait);
    delay(wait);
  } else {
    Serial.printf("[loop] cycle took %lu ms - OVERRAN the %d ms period by "
                  "%lu ms (raise CYCLE_PERIOD_MS)\n",
                  (unsigned long)elapsed, CYCLE_PERIOD_MS,
                  (unsigned long)(elapsed - CYCLE_PERIOD_MS));
  }
}
