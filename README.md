# Project Echo - A bioacoustics classification tool 🤖

We are excited to announce that Project Echo has made significant progress towards our vision of developing an AI/ML solution for understanding the density and classification of noise-producing animals in rainforests. Our goal is to provide conservationists with an efficient and non-destructive tool for tracking threatened animal species population numbers over time.

We have successfully created a complex audio pipeline that can take in any audio file of any type, file size, and format and convert it to tensor Mel spectrogram file ready for training. We have also trained an efficient net_v2 pre-trained model on images with 7.1 million parameters for 224x224 images and attained an impressive accuracy of 90%.

To make our solution more accessible, we have developed a publicly facing pip package that developers can use to load our pre-trained model and pass it raw audio files in a live environment. Additionally, we have developed a prototype API for a website, which can be used to upload a sample audio file and receive predictions for animal classification.

We believe that sound analysis can revolutionize the way we monitor animal populations, and we aim to make our solution widely available to conservationists and researchers. Our solution leverages machine learning to understand the density and classifications of animals in a survey area, using our curated sound datasets. Our team is making solid and comprehensive progress in the analysis of labeled data and its predictive ability using Artificial Intelligence and Machine Learning.

Project Echo is dedicated to providing conservationists with the best possible tools to protect and preserve endangered wildlife and minimize the impact of predatory and other destructive fauna. We believe that sound analysis can unlock new insights into animal populations and are excited to continue our progress towards our vision.

# Our Git File Structure 📂:

![MicrosoftTeams-image (1)](https://user-images.githubusercontent.com/3150898/225461178-079563ee-0f3b-4364-8350-753f4047e82b.png)

Throughout this project we want to try and maintain the above file structure so that we have a clear understanding of where prototype like files live, and developement / component type files.

# Try out our model! - Currently Unavailable...

Install our pip package:
``` python
%pip install git+https://github.com/Deakin-Capstone-Echo/Project-Echo --quiet
```

Pass it your raw audio file and watch the magic happen :)
``` python
import os
import Echo

path_to_raw_audio_file = '/Users/stephankokkas/Downloads/1844_Feldlerche_Gesang_Wind_short.mp3'

my_model = Echo.load_model()
classification = Echo.predict(my_model, path_to_raw_audio_file, traverse_path=False)
```

Our model will use a sliding window of 5 seconds to predict on each 5 second segmont of the audio file you supply! You can therefore expect an output like:
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

*Please note that splitting audio files into 5-second windows may not always be possible in an even manner. To address this issue, we employ an audio padding technique that adds silence to the last window ([-1]th window), which may adversely affect the accuracy of predicting the [-1]th window. We suggest disregarding the [-1]th prediction until this issue is resolved. As evident in the example given above, the [-1]th prediction identifies a white-plumed honeyeater with a confidence level of 62.92% when it should be a skylark.*
