# Engine - Youtube Video to Audio Scraping


## Purpose:
This script allows users to download the audio stream from a specified YouTube video and save it in the WAV audio format to a chosen directory.

## Key Features:

1.  URL Input: Accepts a YouTube video URL to identify the video from which audio will be extracted.
2.  Audio Extraction: Uses the pytube library to fetch and download the audio stream of the given YouTube video.
3.  Audio Conversion: Converts the downloaded audio to the WAV format using the moviepy library.
4.  Custom Output Directory: Allows users to specify a desired directory for saving the WAV file.

## Workflow:

1. The script first checks if the specified output directory exists. If not, it creates the directory.
2. It connects to the YouTube video using the provided URL.
3. The audio stream of the video is identified and downloaded to the specified directory.
4. The downloaded audio is then converted to WAV format.
5. Optionally, the original downloaded audio (before conversion) is deleted to free up space.

## Dependencies:

1. pytube: For downloading video and audio streams from YouTube.
2. moviepy: For audio and video processing, specifically converting audio formats in this script.

## Importance 
This code is very useful for downloading all the vocal audio files needed for project Echo, when training models.


## Sample Video Details 
https://www.youtube.com/watch?v=iy-9Z2KrjsY

## Audio Output
https://drive.google.com/file/d/1SKqE5uINGw53ZIkAWnVCRxgp1uPBCi8o/view?usp=sharing
