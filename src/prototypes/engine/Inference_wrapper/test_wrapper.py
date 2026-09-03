"""
test_wrapper.py

Sprint 1 Prototype

Runs validation tests for the prototype inference wrapper.
"""

from prototype_engine import PrototypeEngine


# -------------------------------------------------
# Sample valid simulator event
# -------------------------------------------------

valid_event = {
    "timestamp": "2026-08-08T10:00:00Z",
    "sensorId": "mic_01",

    "microphoneLLA": [
        -38.143,
        144.361,
        15
    ],

    "animalEstLLA": [
        -38.142,
        144.360,
        15
    ],

    "animalTrueLLA": [
        -38.142,
        144.360,
        15
    ],

    "animalLLAUncertainty": 8.5,

    "audioClip": "VGhpcyBpcyBhIHRlc3QgYXVkaW8="
}


# -------------------------------------------------
# Test 1
# Valid prediction
# -------------------------------------------------

print("=" * 60)
print("TEST 1 - VALID INPUT")
print("=" * 60)

response = PrototypeEngine.process_prediction(
    valid_event,
    "Koala",
    96.42,
    48000
)

print(response.to_dict())

backend_payload = PrototypeEngine.create_backend_payload(response)

print("\nBackend Payload")

print(backend_payload)



# -------------------------------------------------
# Test 2
# Missing sensorId
# -------------------------------------------------

print("\n")
print("=" * 60)
print("TEST 2 - MISSING REQUIRED FIELD")
print("=" * 60)

missing_sensor = valid_event.copy()

del missing_sensor["sensorId"]

response = PrototypeEngine.process_prediction(
    missing_sensor,
    "Koala",
    96.42,
    48000
)

print(response.to_dict())



# -------------------------------------------------
# Test 3
# Invalid audio
# -------------------------------------------------

print("\n")
print("=" * 60)
print("TEST 3 - INVALID AUDIO")
print("=" * 60)

invalid_audio = valid_event.copy()

invalid_audio["audioClip"] = ""

response = PrototypeEngine.process_prediction(
    invalid_audio,
    "Koala",
    96.42,
    48000
)

print(response.to_dict())



# -------------------------------------------------
# Test 4
# Inference failure
# -------------------------------------------------

print("\n")
print("=" * 60)
print("TEST 4 - INFERENCE FAILURE")
print("=" * 60)

response = PrototypeEngine.process_prediction(
    valid_event,
    None,
    None,
    48000
)

print(response.to_dict())