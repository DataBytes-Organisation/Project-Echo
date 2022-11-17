# Bioacoustics-Classification-Tool
Capstone Project (A) - Bioacoustics Classification Tool

Project Description 

Echo Vision 

Echo’s vision is to develop an AI/ML end-to-end solution to understand the density and classification of noise producing animals in a rainforest. This acts to survey the environment in a non-destructive and efficient way to track threatened animal species population numbers over time. 

Project Echo’s impetus stems from the work done by OpenAI: Whisper. Whisper is an automatic speech recognition system which is trained using 680,000 hours of multilingual and multitask supervised data. Whisper takes an input audio file and transcribes it in multiple languages, as well as a translation from one language to another. 

Whispers architecture implements encoder-decoder transformers, which splits the input into 30-second chucks, and converts it into a log-Mel-spectrogram which is passed to the encoder. A decoder is then trained to predict the corresponding text using tokens that direct the model to detect language, transpose and translate. 

We believe there exist many use cases in extending this work to the domain of population density research of any sound generating living or non-living objects. However, to bring life to the project, we will begin by solving the density and classification problem of noise producing animals in rainforests. 

At present, predator density and distribution are monitored by camera grids, GPS collars and scat surveys. While these techniques are effective, they are not without limitations. GPS collars can only be utilized if an animal is caught first and are only effective if they remain on the animal. Camera grids are useful in that they provide visual information that can confirm the presence of an animal as well as their numbers. However, cameras are only effective if the animal/s enter the camera’s field of vision. Scat surveys are an effective way of determining animal movements and numbers, however this process is manual and requires personnel to go out and survey the area. Project Echo will aim to provide an additional tool that utilizes a yet to be utilized modality; Sound. 

Echo Description 

What: 

To provide tools that will aid conservationists in their efforts to protect and preserve endangered wildlife and minimize the impact of predatory and other destructive fauna in line with company values. 

How: 

Build a public library/callable API that can take an audio file input, pass the input to a trained classification model, and return an animal classification with high accuracy. 

The solution will leverage machine learning to understand the density and classifications of animals in a survey area. Using existing sound datasets, our team will work to make solid and comprehensive progress in the analysis of labelled data, and its predictive ability using Artificial Intelligence and Machine Learning.  A frontend application will allow users to record sound samples and location data and upload for processing.  We aim to use existing recorded audio samples to train a model to detect the most likely species contained in a given audio sample with a sufficient level of confidence. 

Here is a test change :)