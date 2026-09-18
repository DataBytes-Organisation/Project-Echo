# Sprint 2 Schema Conformance Report

## Purpose

This report documents the compatibility between the
standard Engine inference response introduced in Sprint 1
and the payload currently sent to the Backend.

## Current Backend Endpoint

POST /engine/event

## Verified Backend Payload Fields

| Field | Engine Response | Backend Payload | Status |
|---|---|---|---|
| timestamp | string | string | PASS |
| species | string / null | string / null | PASS |
| confidence | float / null | float / null | PASS |
| sensorId | string | string | PASS |
| microphoneLLA | GPS object | `[lat, lon, alt]` array | PASS |
| animalEstLLA | GPS object | `[lat, lon, alt]` array | PASS |
| animalTrueLLA | GPS object | `[lat, lon, alt]` array | PASS |
| animalLLAUncertainty | float / null | float / null | PASS |
| audioClip | Base64 string / null | Base64 string / null | PASS |
| sampleRate | integer | integer | PASS |

## GPS Conversion

The standard Engine response represents GPS coordinates as:

```json
{
  "latitude": -38.143,
  "longitude": 144.361,
  "altitude": 15
}
"""
The current Backend payload expects:

[
  -38.143,
  144.361,
  15
]

BackendAdapter performs this conversion.

Evidence

Existing Project Echo event data contains GPS values as
three-element arrays:

"microphoneLLA": [
  -38.78296683180401,
  143.57364908695934,
  10
],
"animalEstLLA": [
  -38.78262719202059,
  143.57271165373092,
  10.074897957321895
],
"animalTrueLLA": [
  -38.7826271920199,
  143.5727116541894,
  10
]

rror Handling

The standard Engine response supports:

{
  "status": "failed",
  "error": {
    "code": "INVALID_AUDIO",
    "message": "Audio clip is empty."
  }
}

The current Backend adapter excludes the status and error
fields from the Backend detection payload. """ 
