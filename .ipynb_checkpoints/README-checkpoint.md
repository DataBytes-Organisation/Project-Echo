# Project Echo - Bioacoustics Classification Tool 🤖

We are thrilled to share the progress of Project Echo, where we aim to develop an AI/ML solution for discerning the density and classification of noise-producing animals in rainforests. Our ultimate goal is to furnish conservationists with an efficient and non-invasive tool for monitoring threatened animal species populations over time.

We firmly believe that sound analysis can revolutionize animal population monitoring. Our solution leverages machine learning techniques to comprehend animal density and classifications within a survey area, utilizing meticulously curated sound datasets.

Project Echo is committed to equipping conservationists with cutting-edge tools to safeguard endangered wildlife and mitigate the impact of predatory and destructive fauna. We are convinced that sound analysis holds the key to unlocking new insights into animal populations, and we are dedicated to advancing towards our vision.

## Key Achievements

We have achieved significant milestones in our journey:

- **Audio Pipeline Development:** We've successfully engineered a sophisticated audio pipeline capable of processing audio files of any type, size, or format. It converts them into tensor Mel spectrogram files, which are essential for training.
  
- **Model Training:** Our efforts have led to the training of an EfficientNetV2 pre-trained model, boasting 7.1 million parameters for 224x224 images. This model has achieved an impressive accuracy of 90%.

- **Accessibility Improvements:** To enhance accessibility, we've developed a publicly available pip package. This package allows developers to load our pre-trained model and analyze raw audio files in real-time. Additionally, we've prototyped an API for a website, enabling users to upload audio samples and receive animal classification predictions.

## Repository File Structure

In order to ensure clarity and ease of locating files within our project, we have established a structured file organization system. This structure has been designed to streamline our workflow and enhance collaboration among team members.

Initially, we have implemented distinct subdirectories for research and documentation purposes. This separation allows for efficient management of project-related materials and facilitates easy access when needed.

Furthermore, within our file structure, we have introduced the 'src' folder, which serves as the repository for our source code. Within the 'src' folder, you will find subdirectories dedicated to components and prototypes. These subdivisions aid in categorizing and managing our source code effectively.

Additionally, within the components and prototypes subdirectories, you will discover individual folders corresponding to the specific components of our project. This granular classification ensures that each aspect of our project is appropriately organized and accessible.

By maintaining this structured file hierarchy, we aim to enhance our productivity and collaboration while minimizing confusion and time wastage in locating essential files.

![Project Echo File Structure](https://github.com/DataBytes-Organisation/Project-Echo/assets/95070150/e8aaf728-c80f-42c1-af07-40651438c002)

## [Work in Progress] Usage

- Install our pip package:

```python
%pip install git+https://github.com/DataBytes-Organisation/Project-Echo --quiet
```

- Use the package to analyze your raw audio files:

```python
import os
import Echo

path_to_raw_audio_file = '/Users/stephankokkas/Downloads/1844_Feldlerche_Gesang_Wind_short.mp3'

my_model = Echo.load_model()
classification = Echo.predict(my_model, path_to_raw_audio_file, traverse_path=False)
```

- Our model applies a sliding window of 5 seconds to predict each segment of your audio file. You can expect an output similar to the following:

```
Your audio file is: 1844_Feldlerche_Gesang_Wind_short.mp3
Your file is split into 11 windows of 5 seconds width per window. For each sliding window, we found:
    A skylark with a confidence of 99.68%
    A skylark with a confidence of 92.1%
    A skylark with a confidence of 99.71%
    A skylark with a confidence of 99.97%
    A skylark with a confidence of 94.82%
    A skylark with a confidence of 99.47%
    A skylark with a confidence of 99.91%
    A skylark with a confidence of 92.84%
    A skylark with a confidence of 99.82%
    A skylark with a confidence of 92.64%
    A white-plumed honeyeater with a confidence of 62.92%
```

> Please note: In some cases, splitting audio files into 5-second windows may not yield even segments. To address this, we use an audio padding technique, adding silence to the last window. This may affect the accuracy of predictions for the last window. Hence, we advise disregarding predictions for the last window until this issue is resolved. As shown in the example above, the prediction for the last window incorrectly identifies a white-plumed honeyeater with a confidence of 62.92% instead of a skylark.

---

*This README document provides an overview of Project Echo, an academic initiative at Deakin University during Trimester 1 of 2024. It outlines the project's goals, structure, evaluation criteria, support mechanisms, and feedback processes.*