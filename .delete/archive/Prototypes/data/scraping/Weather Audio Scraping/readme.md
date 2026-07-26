# Weather Audio Collection

## Task Description

The Weather Audio Collection task aims to gather and curate a comprehensive dataset of audio samples of naturally occurring weather phenomena. These include sounds from rain, thunder, floods, earthquakes, tornadoes, and volcanoes. This dataset is intended to train and enhance weather detection models, aiding in the advancement of automated weather recognition systems.

### Data Collection

Audio samples are collected via a custom script that scrapes relevant YouTube videos, extracts the audio, and stores it in a categorized format. Each weather event type has its own dedicated subdirectory, making the dataset organized and accessible for model training purposes.

### Dataset Structure

The dataset is organized into a compressed file containing subdirectories for each type of weather event. Each subdirectory contains audio samples specific to that weather event. Below is an example of the dataset structure:

weather_sounds.zip/
├── rain/
│ ├── sample1.wav
│ ├── sample2.wav
│ └── ...
├── thunder/
│ ├── sample1.wav
│ ├── sample2.wav
│ └── ...
├── flood/
│ ├── sample1.wav
│ ├── sample2.wav
│ └── ...
└── etc...


### Data Preprocessing and Cleaning

The data preprocessing pipeline involves several steps:
1. **Audio Data Scraping**: Utilizing a script to scrape and download audio from YouTube.
2. **Event Segmentation**: Applying the Yamnet model to segment the audio data based on the event detected in the recordings. This was done using a pre-existing code in the Github Repo. 

Link: https://github.com/DataBytes-Organisation/Project-Echo/tree/main/src/Prototypes/engine/Event_Segmentation_YamNet

3. **Manual Cleaning**: Manually reviewing and cleaning the dataset to ensure the quality and relevance of the audio samples.

### Data Storage Summary

The following table provides an overview of the current dataset, detailing the number of files available for each category of weather event.

| Weather Event | Number of Files   |
|---------------|-------------------|
| Rain          | 4294              |
| Thunder       | 3601              |
| Flood         | 7                 |
| Earthquake    | 82                |
| Tornado       | 2760              |
| Volcano       | 19                |

Link to the Dataset: https://deakin365-my.sharepoint.com/:u:/g/personal/s222523115_deakin_edu_au/EVAS6RrkYfVFrqwnuHDuL8oBDwr1gw_82TMxx_IZHd2GMw?e=PBNZe9

### Definition of Done

The project reaches completion when:
- A compressed file structure with organized subdirectories for each type of weather event is prepared.
- Each subdirectory contains respective weather event audio samples ready for model training.

### How to Contribute

Contributions to the Weather Audio Collection are welcome! Here are ways you can contribute:
- **Data Contribution**: Add more audio samples to our collection.
- **Feature Enhancement**: Suggest or implement improvements to the audio scraping and processing scripts.
- **Bug Reporting**: Report issues or bugs found in the dataset or scripts.
