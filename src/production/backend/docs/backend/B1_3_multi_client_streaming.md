B1.3 Multi-Client Detection Streaming Validation

1. Overview

B1.3 validates the live detection WebSocket behaviour of the Project Echo Backend when multiple HMI clients are connected at the same time.

This task builds on the B1.2 persistence-to-streaming integration already merged into `main`.

The purpose of B1.3 is to verify that:

- multiple authenticated clients can receive the same live detection;
- disconnecting one client does not affect other connected clients;
- a disconnected client can reconnect and receive future detections;
- stale or failed clients do not prevent delivery to healthy clients;
- all connected clients receive a consistent representation of the same persisted event;
- the existing persistence-before-broadcast behaviour remains unchanged.


2. Existing Streaming Flow

B1.3 does not introduce a new streaming architecture. It validates the existing B1.2 implementation.

The live detection flow is:

```text
POST /engine/event
        |
        v
Validate detection payload
        |
        v
Persist detection to MongoDB
        |
        v
Build persisted stream payload
        |
        v
DetectionStreamManager.broadcast()
        |
        +----> WebSocket Client A
        |
        +----> WebSocket Client B
        |
        +----> Additional connected clients

The WebSocket endpoint used by HMI clients is:

ws://localhost:9000/ws/detections?token=<JWT>

A valid JWT must be provided before the connection is accepted.

3. Relevant Backend Components

The validation focuses on the existing components:

app/routers/live.py
app/services/detection_stream.py
app/routers/engine.py
app/routers/live.py

Provides the /ws/detections WebSocket endpoint.

The endpoint:

extracts the JWT from the request;
validates the token;
rejects unauthenticated connections;
registers authenticated clients with the shared detection stream manager;
removes disconnected clients.
app/services/detection_stream.py

DetectionStreamManager maintains the active WebSocket connection set.

The manager provides:

connect()
disconnect()
broadcast()
connection_count

During broadcast, delivery failures are isolated so a stale client does not prevent other clients from receiving the event.

app/routers/engine.py

POST /engine/event first persists the event to MongoDB.

Only after persistence succeeds is the persisted detection representation broadcast to connected WebSocket clients.

This preserves the B1.2 persistence-before-streaming contract.

4. Automated Validation

Automated B1.3 coverage is implemented in:

tests/test_detection_stream_multiclient.py

Four scenarios are tested.

4.1 Simultaneous delivery to two clients

Two clients are connected to the same DetectionStreamManager.

A valid event is submitted through the existing engine ingestion path.

Expected behaviour:

Client A receives event
Client B receives event
Client A payload == Client B payload
connection_count == 2

This verifies that the stream supports multiple simultaneous consumers.

4.2 Disconnect isolation

Two clients receive an initial detection.

Client A is disconnected before a second detection is broadcast.

Expected behaviour:

Client A receives no further events
Client B continues receiving events
connection_count == 1

This confirms that one client leaving does not interrupt the stream for other clients.

4.3 Reconnection

A client disconnects and later reconnects using a new WebSocket connection.

Expected behaviour:

Events produced while offline are not automatically replayed
The reconnected client receives future detections
Existing clients continue receiving detections normally

Historical detections can be retrieved through the existing REST APIs when required.

4.4 Stale-client isolation

A simulated WebSocket raises an exception when the server attempts to send a detection.

Expected behaviour:

Healthy clients still receive the detection
Failed client is removed from the active connection set
Broadcast operation continues successfully

This verifies that one broken connection cannot break delivery for all clients.

5. Automated Test Results

B1.3-specific test command:

python -m pytest -q tests/test_detection_stream_multiclient.py

Result:

4 passed, 1 warning

Regression validation with the existing B1.2 engine ingestion tests:

python -m pytest -q \
    tests/test_engine_ingestion.py \
    tests/test_detection_stream_multiclient.py

Result:

10 passed, 1 warning

The warning is the existing python_multipart deprecation warning and is unrelated to B1.3.

6. Manual Multi-Client Test Tool

A lightweight browser test client was added at:

tools/detection_stream_test_client.html

The tool provides two independent WebSocket clients:

Client A
Client B

Each client supports:

connect;
disconnect;
reconnect;
independent connection status;
received-message count;
latest event ID;
JSON payload log.

The tool also provides a comparison function that verifies whether the latest message received by Client A and Client B is identical.

The tool can be served locally using:

python -m http.server 8085 -d tools

and opened at:

http://localhost:8085/detection_stream_test_client.html

7. Manual Validation

7.1 Scenario 1 — Simultaneous delivery

Both Client A and Client B were connected to:

ws://localhost:9000/ws/detections

A Dingo detection was submitted using:

POST /engine/event

Observed result:

Client A: CONNECTED
Messages received: 1

Client B: CONNECTED
Messages received: 1

Both clients received the same persisted event ID.

The test client's comparison result was:

PASS: Client A and Client B received identical latest payloads.

Result: PASS

7.2 Scenario 2 — Disconnect isolation

Client A was manually disconnected while Client B remained connected.

A second detection for Crimson Rosella was submitted.

Observed result:

Client A:
DISCONNECTED
Messages received: 1

Client B:
CONNECTED
Messages received: 2

Client A did not receive the new detection while offline.

Client B continued receiving detections normally.

Result: PASS

7.3 Scenario 3 — Reconnection

Client A was reconnected after Scenario 2.

A third detection for Sus Scrofa was submitted.

Observed result:

Client A:
CONNECTED
Messages received: 2

Client B:
CONNECTED
Messages received: 3

Both clients received the new Sus Scrofa event.

The latest payload comparison returned:

PASS: Client A and Client B received identical latest payloads.

The different total message counts are expected.

Client A was disconnected during the second detection and therefore did not receive that live event.

Result: PASS

8. Stream Payload Validation

The streamed payload is derived from the persisted detection representation.

Observed fields include:

_id
commonName
type
status
diet
timestamp
sensorId
sourceType
species
microphoneLLA
animalEstLLA
animalTrueLLA
animalLLAUncertainty
confidence

For a given detection, all simultaneously connected clients receive the same payload and event ID.

9. Behaviour Summary
Scenario	Expected	Observed	Result
Two clients connected	Both receive the same detection	A=1, B=1, identical payload	PASS
Client A disconnects	B continues receiving	A=1, B=2	PASS
Client A reconnects	A receives future detections	A=2, B=3	PASS
Stale client failure	Healthy clients still receive	Covered by automated test	PASS
Persistence before broadcast	Persisted event is streamed	Covered by engine integration test	PASS
10. Security and Test Configuration

The manual test tool requires a valid JWT.

The JWT is entered only into the local browser test page and is not hard-coded into the source file.

No JWT, password, .env file, MongoDB credential, or other local secret is included in this documentation or committed to Git.

11. Files Added for B1.3
tests/test_detection_stream_multiclient.py
tools/detection_stream_test_client.html
docs/backend/B1_3_multi_client_streaming.md
12. Conclusion

B1.3 confirms that Project Echo's live detection stream supports multiple simultaneous WebSocket consumers.

The combined automated and manual validation demonstrates that:

multiple clients receive the same persisted detection;
one client can disconnect without disrupting others;
reconnecting clients receive subsequent live events;
stale connections are isolated and removed;
event payloads remain consistent across connected clients;
the B1.2 persistence-before-streaming behaviour remains intact.

Therefore, the multi-client detection streaming behaviour required by B1.3 has been successfully validated.