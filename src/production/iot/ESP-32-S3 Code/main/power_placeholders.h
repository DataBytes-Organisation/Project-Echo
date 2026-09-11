#pragma once
#include <Arduino.h>

// ===========================================================================
// power_placeholders.h
//
// Stubs so the sketch compiles and runs while the power and cellular work is
// still owned by someone else. Each one prints and returns success.
//
// Keep the signatures stable so swapping in the real implementations is a
// drop-in change and nothing in main.ino has to move.
// ===========================================================================

// DFRobot solar / LiPo charger init.
// Real version should: bring up I2C, read battery voltage and charge state,
// and probably refuse to record if the pack is below some cutoff.
static inline bool connectSolar() {
  Serial.println("[power] connectSolar() placeholder - no solar hardware yet");
  return true;
}

// Cat-1 4G module init. WiFi is used for the MVP.
// Real version should: power the modem, wait for network registration,
// bring up PDP context, and expose a Client* that PubSubClient can use
// instead of WiFiClient.
static inline bool connect4G() {
  Serial.println("[power] connect4G() placeholder - using WiFi for now");
  return true;
}

// Returns battery percentage, or -1 when unknown.
static inline int batteryPercent() {
  return -1;
}

// Real version should configure the wake source and call
// esp_deep_sleep_start(). Until then this is just a blocking delay so the
// loop timing matches what the deep-sleep version will do.
static inline void sleepFor(uint32_t ms) {
  Serial.printf("[power] sleepFor(%lu) placeholder - plain delay, not deep sleep\n",
                (unsigned long)ms);
  delay(ms);
}
