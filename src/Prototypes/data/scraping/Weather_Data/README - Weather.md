# Engine - Data collection of Weather audio

## Overview 
This database contains cleaned audio recordings of various Weather data such as the Earthquake, Rain, Thunder & Wind. This data is classified into folders.
The audio files are scraped from Youtube using Youtube-video-to-audio scraping code and then event segmentation (Event Segmenter) was performed on each of the audio files to identify event(audio) and saved in a folder post whcih the data was cleaned and stored for future training purpose. 

# Youtube_Video_to_Audio scraper used:
https://github.com/DataBytes-Organisation/Project-Echo/blob/main/src/Prototypes/data/scraping/Youtube%20video%20to%20audio%20(.wav%20format)/README.md

## Details of the Data Bucket
- **Total Size**: 10.8 GB MB
- **Number of files**: 16,986 Files 
- **Directory Structure**: The audio files are stored in 'Weather.zip' folder, which contains 4 subfolders like 'Earthquake', 'Rain', 'Rainy & Thunder', 'Windy & Thunder' and 'Wind'.

## Importance 
These audio files are of crucial importance to project Echo, when training models related to weather arrangements.


## Sample Audio Details 

| File Name             | Audio Samples |
|-----------------------|---------------|
| Earthquake            |     2739      |
| Rain                  |     5664      |
| Rainy & Thunder       |     3601      |
| Windy & Thunder       |     2284      |
| Wind                  |     2698      |



## Play the Sample audio files 

https://deakin365-my.sharepoint.com/:u:/g/personal/s222502507_deakin_edu_au/EWavqfiFfU5Di-TIDEZdaNAB5gwb4FdLdsr3peO2qChWVA?e=Lt4NUy

Access this link to play and download the sample audios.



