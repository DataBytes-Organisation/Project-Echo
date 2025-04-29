# Project Echo - Bioacoustics Classification Tool 🤖

We are thrilled to share the progress of Project Echo, where we aim to develop an AI/ML solution for discerning the density and classification of noise-producing animals in rainforests. Our ultimate goal is to furnish conservationists with an efficient and non-invasive tool for monitoring threatened animal species populations over time.

We firmly believe that sound analysis can revolutionize animal population monitoring. Our solution leverages machine learning techniques to comprehend animal density and classifications within a survey area, utilizing meticulously curated sound datasets.

Project Echo is committed to equipping conservationists with cutting-edge tools to safeguard endangered wildlife and mitigate the impact of predatory and destructive fauna. We are convinced that sound analysis holds the key to unlocking new insights into animal populations, and we are dedicated to advancing towards our vision.

---

## Key Achievements

We have achieved significant milestones in our journey:

- **Audio Pipeline Development:** We've successfully engineered a sophisticated audio pipeline capable of processing audio files of any type, size, or format. It converts them into tensor Mel spectrogram files, which are essential for training.
  
- **Model Training:** Our efforts have led to the training of an EfficientNetV2 pre-trained model, boasting 7.1 million parameters for 224x224 images. This model has achieved an impressive accuracy of 90%.

- **Accessibility Improvements:** To enhance accessibility, we've developed a publicly available pip package. This package allows developers to load our pre-trained model and analyze raw audio files in real-time. Additionally, we've prototyped an API for a website, enabling users to upload audio samples and receive animal classification predictions.

---

## Repository File Structure

In order to ensure clarity and ease of locating files within our project, we have established a structured file organization system. This structure has been designed to streamline our workflow and enhance collaboration among team members.

Initially, we have implemented distinct subdirectories for research and documentation purposes. This separation allows for efficient management of project-related materials and facilitates easy access when needed.

Furthermore, within our file structure, we have introduced the 'src' folder, which serves as the repository for our source code. Within the 'src' folder, you will find subdirectories dedicated to components and prototypes. These subdivisions aid in categorizing and managing our source code effectively.

Additionally, within the components and prototypes subdirectories, you will discover individual folders corresponding to the specific components of our project. This granular classification ensures that each aspect of our project is appropriately organized and accessible.

By maintaining this structured file hierarchy, we aim to enhance our productivity and collaboration while minimizing confusion and time wastage in locating essential files.

![Project Echo File Structure](https://github.com/DataBytes-Organisation/Project-Echo/assets/95070150/e8aaf728-c80f-42c1-af07-40651438c002)

## Installation Guide

### Prerequisites
Before proceeding, ensure the following dependencies are installed on your system:

- **Python**: Version 3.7 or later.
- **ffmpeg**: Required for audio processing.
  - **macOS**: `brew install ffmpeg`
  - **Windows**: `choco install ffmpeg`
  - **Linux**: `sudo apt update && sudo apt install ffmpeg`
- **pip**: Updated to the latest version:

```bash
 pip install --upgrade pip
```

### Dependency Table

The following are the required dependencies and their compatible versions:

| Dependency          | Version Range          | Notes                                   |
|---------------------|------------------------|-----------------------------------------|
| TensorFlow          | >=2.10.0,<2.15.0      | Use `tensorflow-cpu` for systems without AVX support |
| NumPy               | >=1.21.0,<1.25.0      | Conflicts may arise with `librosa` or `tensorflow` |
| Keras               | >=2.10.0,<2.11.0      | Matches TensorFlow version requirements |
| Protobuf            | >=3.19.6,<3.20.0      | Used for TensorFlow and related packages |
| ffmpeg-python       | 0.2.0                 | Required for audio processing           |
| SciPy               | >=1.7.0               | Used for numerical and scientific computing |
| Numba               | >=0.56.0              | Required by `librosa`                  |
| Librosa             | 0.9.2                 | For audio analysis and feature extraction |
| Typing-Extensions   | >=4.1.1,<5.0.0        | May conflict with older dependencies    |


### Installation
To install the Project Echo pip package, use the following command:
```bash
pip install git+https://github.com/DataBytes-Organisation/Project-Echo --quiet
```

This will install the latest version of the package along with its dependencies.


## Troubleshooting Installation Issues

If you encounter dependency conflicts or errors during the installation process, consider the following:

1. **Use a Virtual Environment:** Create a clean virtual environment to isolate dependencies:

```bash
conda create -n project-echo-env python=3.9
conda activate project-echo-env
```

2. **Install CPU-Compatible TensorFlow:** For systems without AVX support, install the CPU-compatible TensorFlow version

```bash
pip install tensorflow-cpu==2.10.0
```

3. **Resolve Dependency Conflicts:** Use pipdeptree to identify dependency conflicts:

```bash
pip install pipdeptree
pipdeptree --warn fail
```

4. **Manually Install Compatible Versions:**

```bash
pip install numpy==1.24.4 keras==2.10.0 protobuf==3.19.6

```


## Usage

**Single File Prediction**

Once installed, you can analyse a single raw audio file as follows:

python code:

```bash
import os
import Echo

path_to_raw_audio_file = '/path/to/your/audio_file.mp3'

my_model = Echo.load_model()
classification = Echo.predict(my_model, path_to_raw_audio_file, traverse_path=False)

print(classification)
```

### Directory-Wide Prediction
To analyze all audio files in a directory, use:

```bash
path_to_audio_directory = '/path/to/your/audio_directory/'

classification = Echo.predict(my_model, path_to_audio_directory, traverse_path=True)
print(classification)
```

### Example output:

```bash
Your audio file is: audio_file.mp3
Your file is split into 11 windows of 5 seconds width per window. For each sliding window, we found:
    A skylark with a confidence of 99.68%
    A skylark with a confidence of 92.1%
    ...
    A white-plumed honeyeater with a confidence of 62.92%
```

- **Note**: The model applies a sliding window of 5 seconds to analyze each segment of your audio file. Predictions for the last segment may be less accurate due to padding, as shown in the example above.

## Known Issues

### Dependency Conflicts
- Some dependencies require specific versions:

    - Example: numpy>=1.20.3 is required by librosa but conflicts with tensorflow-cpu.
    
    - Example: typing-extensions>=4.1.1 conflicts with older packages.
    
#### Platform-Specific Notes

- **macOS**:
    - TensorFlow may fail due to AVX compatibility issues on older CPUs. Install the CPU-compatible version using:
        ```bash
        pip install tensorflow-cpu==2.10.0
        ```

- **Windows**:
    - Ensure `choco` is installed for `ffmpeg` installation or download it manually from the official [FFmpeg website](https://ffmpeg.org/).
    - TensorFlow might not fully support certain older Windows setups. Check the [TensorFlow for Windows](https://www.tensorflow.org/install/pip) guide for more details.

- **Linux**:
    - Use the latest pip version to avoid conflicts:
        ```bash
        sudo apt update && sudo apt install python3-pip
        pip install --upgrade pip
        ```

### TensorFlow Compatibility
- The TensorFlow library requires AVX instructions for optimal performance. Use tensorflow-cpu on systems without AVX.

### Behavioural Issues
- Last-window predictions may have inaccuracies due to padding.


## Proposed Fixes (Planned Updates)

- Standardize Installation Instructions: Consolidate all instructions to use the correct pip installation path.
- Add a Custom Pip Command: Implement a custom pip installation script to automate dependency resolution and streamline user experience.

### Proposed Fixes (Planned Updates)

**Custom Pip Command for Simplified Installation**:
    We plan to create a custom pip installation script to automate dependency resolution and streamline the user experience. Once completed, users will be able to install the Project Echo package and all required dependencies using a single command, such as:

```bash
    pip install project-echo
```

This will reduce errors caused by manual installation of conflicting dependency versions and improve usability for contributors and end-users.



## **Update `requirements.txt`:**
  ```plaintext
  numpy>=1.21.0,<1.25.0
  ffmpeg-python==0.2.0
  tensorflow>=2.10.0,<2.15.0
  keras>=2.10.0
  scipy>=1.7.0
  numba>=0.56.0

```
## Repository File Structure

We have organized the repository to streamline collaboration and ease of access:

- **`Design/`**: Contains design documents and diagrams.
- **`Echo/`**: The core package implementation.
- **`Requirements/`**: Dependency-related files (`requirements.txt`, `setup.py`).
- **`Research/`**: Reports, datasets, and experiment results.
- **`src/`**: Source code for various project components.
- **`Tutorials/`**: Example usage notebooks.

---

*This README document provides an overview of Project Echo, an academic initiative at Deakin University during Trimester 1 of 2024. It outlines the project's goals, structure, evaluation criteria, support mechanisms, and feedback processes.*


