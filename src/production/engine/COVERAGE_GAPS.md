\# Remaining Engine Coverage Gaps



The Sprint 1 automated test suite covers 46% of

`echo\_engine\_iot.py`. The remaining statements mainly involve

external services, legacy audio modes and model workflows requiring

additional assets.



\## Real Model Integration



Model-loading behaviour is tested using mocks. Real integration

testing requires complete Keras, YAMNet or production model files.



The supplied Engine code does not contain a TensorFlow Lite

Interpreter. Testing the selected TFLite production model requires

coordination with the model-integration team member.



\## Additional Audio Workflows



The standard Recording Mode preprocessing path is tested. These paths

remain for future testing:



\- Recording Mode V2

\- Animal simulation mode

\- Weather-audio prediction

\- Sound-event segmentation

\- Multi-segment audio processing



\## External Integrations



These services are mocked in Sprint 1:



\- Google Cloud Storage

\- MongoDB

\- TensorFlow Serving

\- Backend API

\- MQTT brokers



Testing them requires approved development services, credentials and

stable interface contracts.



\## Live MQTT Testing



`test\_iot\_publisher.py` connects to a public HiveMQ broker by default.

It was not included in the automated execution. Future integration

testing should use a local or private development broker.



\## Production Alignment



The automated suite currently targets `echo\_engine\_iot.py`. The team

must confirm when its IoT functionality will be merged into the main

`echo\_engine.py` production file.



\## Species Configuration



`class\_names.json` is present but is not read by the supplied Engine

scripts. The Engine currently uses pickle files and species names from

Google Cloud Storage. The intended production source should be

confirmed.



\## Sprint 2 Recommendations



1\. Add an approved small model fixture.

2\. Add a local Docker-based MQTT test broker.

3\. Test TensorFlow Serving timeouts and invalid responses.

4\. Test Backend errors and retry behaviour.

5\. Add Recording Mode V2 and weather-audio fixtures.

6\. Confirm the production species mapping.

7\. Repeat the suite against the final production Engine file.
