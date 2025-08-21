
Project Echo – Extend Species Audio | Animal Sound Classification (EfficientAT)

Project Contents

This repository includes the following core components:

- Animal Sounds Tracker
  A structured Excel spreadsheet used to log and manage the collection of animal audio samples across species. It helps maintain dataset coverage and species balance during training.

- EfficientAT Binary Classifier
  A PyTorch implementation of a binary classifier using EfficientAT (mn40_as_ext) to distinguish between animal and non-animal environmental sounds. This replaces the previous EfficientNetV2-based model, achieving improved accuracy and scalability.

- Preprocessing Scripts
  Utilities for:
  - Cleaning non-animal datasets (ESC-50, UrbanSound8K)
  - Converting .wav/.mp3 audio into 224×224 mel spectrograms (.pt files)
  - Augmenting training data using time/frequency masking and Gaussian noise 

- Training Pipeline
  A reproducible script for training the EfficientAT-based model, with clear dataset splitting, augmentation toggles, and evaluation metrics (accuracy, F1-score, confusion matrix).

Sound Dataset Download

You can download the latest animal sound files from the following public Google Drive folder:
Download Folder: https://drive.google.com/drive/folders/1MP1j_oiMGL6hWWMrPcuJYsKLSUH8gjp_?usp=sharing

These files contain newly added species recordings and should be added to the data_files directory within the repository:
/src/Prototypes/data/data_files/

The project also supports environmental sounds sourced from:
- ESC-50 GitHub: https://github.com/karoldvl/ESC-50
- UrbanSound8K (via Zenodo): https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz


Model Summary

- Architecture: EfficientAT (mn40_as_ext), pretrained on AudioSet
- Input: 224×224 mel spectrograms
- Training Epochs: 5
- Accuracy:
  - Train: 98.63%
  - Validation: 98.83%
- Final F1-score: 0.99
- Confusion Matrix: Excellent separation between classes, with minimal misclassification.

Authors

- Riley Beckett
- Irfan Boenardi
