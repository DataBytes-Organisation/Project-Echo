Project Echo - Audio Classification for Environmental and Animal Sounds

Overview:
---------
Project Echo is a lightweight and efficient audio classification system designed to identify key environmental sounds (fire, rain, thunder, wind) and animal calls. It uses a pre-trained EfficientAT model (mn40_as_ext) to provide robust multi-label predictions, even under overlapping sound conditions. The model is designed to support wildlife monitoring efforts in natural habitats like the Otways.



Key Features:
-------------
- EfficientAT CNN backbone (mn40_as_ext, pre-trained on AudioSet)
- Fine-tuning only the classifier head
- Multi-label support for overlapping sound events
- Realistic augmentation: noise, pitch/time shifts, time masking, overlap simulation
- Performance metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

Dataset:
--------
- Total samples: 3254 (including both animal and environmental sounds)
- Format: Mel-spectrograms saved as .pt files
- Sampling rate: 16kHz
- Structure:
    - animal/
    - environment/fire/
    - environment/rain/
    - environment/thunder/
    - environment/wind/

Model Source:
-------------
EfficientAT GitHub Repo: https://github.com/fschmid56/EfficientAT

Training Setup:
---------------
- Learning rate: 0.0001
- Weight decay: 0.0001
- Dropout: 0.3
- Overlap probability: 0.1
- Epochs: 5
- Batch size: 8

Results:
--------
- Final Validation Accuracy: 93.46%
- Animal Class Accuracy: 94.46%
- Environmental Class Accuracies:
    - Fire: 85.51%
    - Rain: 56.58%
    - Thunder: 64.63%
    - Wind: 51.85%
- Macro F1 Score: 0.7003
- Weighted F1 Score: 0.8065

Instructions:
-------------
1. Clone the EfficientAT repo and place it in your project directory.
2. Download the preprocessed mel spectrogram dataset from:
   https://drive.google.com/drive/u/2/folders/1hF-x9Mxoj_lWh-j2NQQwythWR_12Hd0F
3. Adjust paths in the Python script as needed.
4. Run the training script (`main.py` or notebook).
5. Evaluate model and visualize confusion matrix.


---------------------------------------
Project Developed By
RILEY BECKETT, VENESH KANAGASINGAM and IRFAN BOENARDI