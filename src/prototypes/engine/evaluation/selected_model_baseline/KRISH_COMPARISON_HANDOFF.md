# EfficientNetV2 Pipeline Parity Check

## Reference artifact verification

The supplied reference artifacts match commit `e590016` on
`EE/KF/main_engine_tflite_integration`:

- `efficientnetv2_predictor.py` Git blob:
  `2df0cad25e9232ad9076bd33adc89ed8c9c584e6`
- `validate_efficientnetv2_tflite_inference.py` Git blob:
  `f9f8c61409e830036a55b959f86f4f9197eb34ee`
- TFLite model Git blob:
  `2059399c36d3e3bc2b940f7d0678d3eb1291af2a`
- TFLite model SHA-256:
  `81169b254b9adfb5ebf0a2651b0b4a2e0fefdc0f857b314bacbfa57ddde2b50f`
- Class mapping SHA-256:
  `b4b453faf511291013e9c5435efd1b86fc7daeff1f883c095e511c68dfe30a8e`
- Preprocessing configuration SHA-256:
  `2fe1aece4c66b85d34c8feedf0b0b60ae8da6deacdc1ba283901a7d9f19cb882`

## Comparison procedure

Run the main Engine pipeline against the audio files in
`Project_Echo_Balanced_Validation_Dataset.zip`. Use
`balanced_validation_manifest.csv` as the fixed file list and apply the exact
preprocessing configuration.

Export one row per file with:

- `relative_path` or `sha256`
- predicted class index and label
- confidence
- preprocessing or inference error, if any

Join the export with `per_file_predictions.csv` using `relative_path` or
`sha256`. Compare predicted index, predicted label, confidence and file
failures. Both pipelines must use the same file list and preprocessing settings.

Expected TFLite interface:

- input: `[1, 1, 128, 313]`, float32
- output: `[1, 123]`, float32 logits
