# Echo Engine Automated Test Instructions

## Purpose

This suite provides lightweight and repeatable automated testing for the consolidated `echo_engine.py`. It combines the existing Engine regression tests with Sprint 2 coverage of the integration path from a standard real-device or simulator MQTT message, through audio decoding, preprocessing and EfficientNetV2 inference, to construction and submission of the Backend event.

The tests run offline. They do not require a live MQTT broker, IoT hardware, TensorFlow Serving, a production model file, the Backend, MongoDB or the HMI.

## Verified environment

- Operating system: Windows
- Conda environment: `projectecho`
- Python: 3.9.25
- Pytest: 8.4.2
- pytest-cov / Coverage.py: 7.1.0 / 7.10.7
- Sprint 2 branch: `EE/SMKN/End-to-End-Automated-Engine-Integration-Testing`
- Upstream baseline: `09f24bb9`

## Setup

Open Anaconda Prompt and run:

```bat
conda activate projectecho
cd /d D:\Project-Echo\src\production\engine
python -m pip install pytest pytest-cov
```

No credentials or production services are required for the selected automated suite.

## Run the Sprint 2 end-to-end tests

```bat
python -m pytest test_engine_end_to_end.py -v --tb=short
```

Verified result on 13 September 2026: **8 passed in 0.53 seconds**.

## Run the input-contract and delivery tests

```bat
python -m pytest test_engine_input_contract.py test_engine_delivery.py -v --tb=short
```

Verified result on 13 September 2026: **27 passed in 1.53 seconds**.

## Run the complete selected Engine suite

```bat
python -m pytest test_iot_integration.py test_engine_core.py test_engine_configuration.py test_engine_model_loading.py test_engine_preprocessing.py test_engine_output.py test_engine_input_contract.py test_engine_delivery.py test_engine_end_to_end.py -q --tb=short
```

The selected suite contains 84 tests. `test_iot_publisher.py` is intentionally excluded because it performs live MQTT publishing and is not part of the reproducible offline run.

## Generate coverage

```bat
python -m pytest test_iot_integration.py test_engine_core.py test_engine_configuration.py test_engine_model_loading.py test_engine_preprocessing.py test_engine_output.py test_engine_input_contract.py test_engine_delivery.py test_engine_end_to_end.py -q --cov=echo_engine --cov-report=term-missing --cov-report=html
```

To display the exact percentage and missing line ranges:

```bat
python -m coverage report -m --precision=2 echo_engine.py
```

To open the HTML report:

```bat
start "" htmlcov\index.html
```

Verified coverage on 13 September 2026: **58.44%** of `echo_engine.py`.

## What the Sprint 2 file tests

`test_engine_end_to_end.py` keeps the important internal Engine stages connected. It uses a small generated WAV fixture and exercises real Engine methods for Base64 decoding, fixed-length audio preparation, mel-spectrogram orchestration, tensor normalisation, TFLite invocation, softmax/class mapping and Backend-event construction.

Only external or expensive boundaries are replaced: DSP calculations return deterministic fixture arrays, the TFLite runtime returns controlled logits, and HTTP requests use a mock Backend response. This keeps the tests fast and reproducible while still checking the complete Engine orchestration path.

## Interpreting failures

- A preprocessing failure must prevent inference and Backend submission.
- An inference or invalid-output failure must prevent Backend submission.
- A Backend timeout is retried only to the configured limit.
- After a failed message, the handler must remain able to process the next message.
- Contract assertion failures may indicate that the agreed Engine-to-Backend schema has changed and should be reviewed before changing the test.
