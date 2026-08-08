# Tests

Project Echo unit, integration, and system tests.

## Integration tests

- `integration/engine_backend/integration_harness/`: Engine-to-Backend request contract, response handling, timeout, connection-failure, and validation coverage.

## Pipeline tests

- `pipeline/engine_training/smoke_test/`: end-to-end smoke test for the PyTorch training pipeline at `src/prototypes/engine/augmentation/` - runs the real `main.py` CLI against a synthetic dataset and checks it produces a checkpoint, TensorBoard log, and class list.
