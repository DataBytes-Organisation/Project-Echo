# Class Activation Mapping (CAM) for Spectrograms
This repository contains a notebook that demonstrates the process of adding Class Activation Mapping (CAM) to spectrograms. The purpose of CAM is to provide visual explanations of which regions in the image were pivotal in making a prediction, making it an essential tool for model interpretability.

### Overview
- Source: This notebook is a modified version of the code found on Kaggle. The original code can be accessed [here.]https://www.kaggle.com/code/guglielmocamporese/what-is-my-model-seeing-class-activation-mapping

- Data: The spectrograms are generated from audio files of three bird species:

1. Charadrius ruficapillus (Red-capped plover)
2. Chenonetta jubata (Australian Wood Duck)
3. Chlamydera nuchalis (Great bowerbird)
The audio data was sourced from Project Echo bucket 2.

### ResNet (Residual Network) Model
ResNet, short for Residual Networks, is a classic neural network used in many deep learning applications, particularly for image classification. It introduces "skip connections" that bypass one or more layers, addressing the vanishing gradient problem in deep networks.

## Getting Started
### Prerequisites
Ensure you have the following libraries installed:

- PIL (Pillow)
- os
- matplotlib
- torch
- torchvision
- numpy
- skimage
- IPython
### Usage

- Update the audio_folder_path and image_folder_path in the notebook to point to your directories.
- Run the notebook to visualize CAM on spectrograms.
### Code Explanations
**Data Preprocessing:** Images are resized to 224x224, converted to tensors, and normalized using ImageNet statistics.

**Loading the ResNet Model:** Loads the pretrained ResNet-18 model for image classification tasks.

**Extracting Features from the Model:** The SaveFeatures class captures the output features from a specified model layer. These features are crucial for computing the CAM.

**Computing the Class Activation Map:** The getCAM function computes the CAM for a given class using features from the convolutional layer and weights from the fully connected layer.

**Applying CAM to Images and Visualization:** After computing the CAM for each image, the notebook overlays the CAM on the images, highlighting the significant regions responsible for the model's prediction.

## Results
![Class Activation Maps and Overlays](<download (4).png>)