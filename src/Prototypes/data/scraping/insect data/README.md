
# Audio Data Bucket: Insect Sounds

## Overview
This database contains audio recordings of insect sounds, specifically from four types of insects: bumblebee, cicada, cricket, and grasshopper.

## Database Details
- **Total Size**: 26.7 MB
- **Number of Files**: 38
- **Unique Authors**: 34
- **Directory Structure**: All audio files are stored in an 'insect' folder, which contains four subfolders: 'bumblebee', 'cicada', 'cricket', and 'grasshopper'.

## Importance
These audio samples are crucial for various audio processing tasks and analyses related to insect sounds.

## Sources
All sounds were sourced from [Freesound.org](http://freesound.org).

## Sample Audio Details
Here are some sample details from the audio dataset:

| File Name | Author | File Length |
|-----------|--------|-------------|
| bumblebee | reinsamba | 26s |
| cicada | InspectorJ | 45s |
| Cicada3 | azdipu | 1min 4s |
| cricket | soundscalpel.co | 1min 17s |
| cricket2 | acclivity | 21s |
| grasshopper2 | bruno.auzet | 3min 21s  |
... (and more)

## Data Cleaning and Preprocessing
The following steps were taken to ensure the quality and consistency of the audio data:
1. **Manual Filtering**: Audio files were manually screened to select those with clearer insect sounds. Files with excessive noise or indistinct insect sounds were filtered out.
2. **Noise Reduction with Audacity**: [Audacity](https://www.audacityteam.org/) was used for noise reduction, and manual editing was done to remove silent or excessively noisy segments.
3. **Loudness Normalization**: Audacity was also used to normalize the loudness of the audio files, ensuring consistency and suitability for training models.

## Sample Spectrogram
Here is the [spectrogram](<https://drive.google.com/file/d/1CyYZlgPwIxO4CzeInysqrA3d5hGB7g1n/view?usp=drive_link>) for the sample audio `bumblebee4.mp3`.

