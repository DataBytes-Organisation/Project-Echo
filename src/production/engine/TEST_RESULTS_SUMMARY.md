# Sprint 2 Engine Test Results Summary

## Final verified results

- Branch: `EE/SMKN/End-to-End-Automated-Engine-Integration-Testing`
- Upstream baseline: `09f24bb9`
- Test date: 13 September 2026
- Python: 3.9.25
- Pytest: 8.4.2
- Tests collected: 84
- Tests passed: 84
- Tests failed: 0
- Engine statements: 628
- Covered statements: 367
- Missing statements: 261
- Coverage of `echo_engine.py`: 58.44%

The complete selected suite ran without live production services or the full audio dataset. The HTML coverage report was generated in `htmlcov/index.html`.

## My Sprint 2 contribution

I added `test_engine_end_to_end.py`, containing eight mocked-boundary integration tests. These tests extend the Sprint 1 foundation by keeping the internal preprocessing and inference stages connected instead of mocking the entire prediction function.

The eight tests verify:

1. A standard real-device event reaches the Backend through the real Engine preprocessing and inference methods.
2. A standard simulator event follows the same complete path while preserving `sourceType`.
3. A legacy Recording Mode event is normalised and reaches the standard Backend contract.
4. An audio-preprocessing failure stops inference and Backend submission.
5. A local-inference failure stops Backend submission.
6. An invalid model-output shape is rejected.
7. A missing winning-class label is rejected.
8. Backend timeout retries are bounded, the error is caught by the handler, and the next message can still be processed.

The focused Sprint 2 file completed **8 tests in 0.53 seconds**.

## Combined regression and integration coverage

The final run also included the existing configuration, model-loading, preprocessing, output, core and IoT regression tests. The upstream input-contract and Backend-delivery suites contributed 27 tests covering schema normalisation, API authentication, HTTP status handling, configured timeouts and bounded retry behaviour.

My end-to-end file complements those suites by testing the connected path from generated WAV audio to the exact HTTP-boundary payload. It does not duplicate every input-validation or HTTP-status case already covered upstream.

## Verified outcomes

- Standard `real` and `simulator` events use the agreed structured LLA fields.
- `sourceType`, sample rate and active model source are preserved in the Backend event.
- Audio is decoded, padded, converted to a mel representation, normalised and supplied to the TFLite interface.
- The predicted species and confidence are placed in the Backend payload.
- Invalid preprocessing or inference results do not produce a Backend request.
- Backend timeouts cannot retry indefinitely.
- One failed event does not prevent processing of the next event.

## Conclusion

All 84 selected tests passed, and the consolidated Engine achieved 58.44% statement coverage. This provides a reproducible automated integration baseline for Sprint 2. Live cross-service validation remains separate and must be coordinated with the teammate responsible for real-flow testing.
