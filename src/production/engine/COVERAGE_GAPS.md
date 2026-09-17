# Remaining Engine Coverage Gaps

## Current position

The selected Sprint 2 suite passes 84 tests and covers **58.44%** of the consolidated `echo_engine.py`: 367 of 628 statements are executed, with 261 statements remaining uncovered.

The percentage should not be compared directly with Sprint 1 coverage of `echo_engine_iot.py`, because Sprint 2 targets a different consolidated module and the production code and test suites have both changed.

## Uncovered or partially covered areas

### Production startup and model assets

The constructor paths for real credentials, MongoDB setup and complete EfficientNetV2 TFLite model initialisation remain only partially covered. Reliable automated execution would require controlled credential files, a small approved model artifact and additional startup fixtures.

Relevant missing ranges include lines 141–142 and 175–288.

### External species and storage services

The Google Cloud species-list workflow is not executed because the offline suite does not use cloud credentials or a live storage bucket.

Relevant missing range: lines 308–323.

### Legacy and secondary audio workflows

The suite covers the current EfficientNetV2 path but does not fully execute the older combined TensorFlow pipeline, stereo-audio branch, `Recording_Mode_V2`, multi-segment sound-event detection, YAMNet feature extraction or weather-audio prediction.

Relevant missing ranges include lines 368–442, 955–1009 and 1199–1342.

These paths require additional model assets and should be confirmed as active production requirements before substantial test effort is added.

### Additional preprocessing branches

The generated WAV fixture covers short-audio padding and NCHW tensor creation. Remaining branches include an uninitialised interpreter/configuration, long-audio truncation and alternative NHWC tensor layout or shape mismatch handling.

Relevant missing lines include 504, 509, 553 and 599–607.

### Input-boundary edge cases

The upstream contract suite covers the main standard and legacy payload shapes. Some lower-frequency validation branches remain, including additional invalid LLA types and values, source-type combinations and legacy-detection conditions.

Relevant missing ranges include lines 783–895.

### Backend delivery edge cases

The existing delivery suite covers success, authentication, timeouts, connection failures, HTTP 400/422 and retryable 5xx responses. Unexpected status categories outside these groups remain uncovered.

Relevant missing range: lines 1167–1174.

### Live MQTT and long-running behaviour

The automated suite calls handlers directly and therefore does not prove broker connectivity, subscription behaviour, disconnect/reconnection timing, continuous listener stability or process startup. The live publisher is intentionally excluded from the offline suite.

Relevant missing ranges include lines 1365–1383, 1501–1516, 1533–1534 and 1554–1555.

### Real cross-service acceptance

HTTP calls are mocked, so the suite does not prove that a live Backend returns HTTP 201, writes the event to MongoDB or exposes it correctly in the HMI. It also does not validate a real ESP32 device, network delays, deployed authentication or production model accuracy.

## Coordination boundary

My automated work should remain focused on fast, reproducible Engine orchestration and failure-path tests. I need to coordinate with Krish so that his real-flow scenarios and my automated scenarios have a documented division and do not duplicate effort. Any live MQTT, Backend, database or HMI evidence should only be claimed after it has actually been completed and recorded.

## Recommended next actions

1. Confirm the automated-versus-real-flow division with Krish and retain the discussion as collaboration evidence.
2. If assigned, run the agreed standard real and simulator fixtures through a development MQTT broker and Backend.
3. Capture evidence at the MQTT input, Engine log, HTTP 201 response, stored Backend event and HMI display.
4. Add automated tests only for high-risk uncovered paths that are still active and are not already covered by another team member.
5. Rerun the selected suite and update this document whenever `echo_engine.py` or the integration contract changes.
