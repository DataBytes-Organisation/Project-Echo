## Bioacoustics-Classification-Tool

### Capstone Project (A) - Bioacoustics Classification Tool

#### Architectural design

This architectural design is intended to elicit the system components and provide an initial view of the overall product design and the lower level data processing and model training pipelines.  It is expected that during implementation, this design will be evolved iteratively as we learn more about how audio processing is performed.

The conceptual product layout has been inspired by previous work using a grid of camera (image data) installations in the natural habitats of endangered species and their predators to learn more about their behaviours and the distribution of population in the environment.  In Project echo, this is extended to focus on the audio domain, rather than images - with the intent that an array of audio sensors would be distributed to capture real time information on species movements.  Information collected via sensors in the field provides valuable insight to researchers and conservationists that can help them more efficiently design and manage preservation mechanisms.

The project is to use AI/ML technology to automate the process of classifying animal species captured by such an array of audio sensors.  As these sensors will be recording 24/7, it is time and cost prohibitive to manually process all such collected data - so an automated approach is required  Project Echo aims to build the audio engine to do this audio classification task automatically in real time.

The basis of this material for audio processing pipeline was derived from the initial literature review material captured in this repository.  There are essentially two approaches that appear to proliferate the research in this space.  The first approach involves processing the audio waveform directly and translating that into a classification result.  The 2nd approach involves first converting the audio into image domain (via something called a Mel-Spectrogram) and them perform classification on the images using the well established image classification paridigm.  The 2nd approach that leverages mel-spectrogram intermediate images has been adopted for the Echo Prototype investigation for 2022-T3.

#### Product Concept Design

The following diagram shows the product concept design the project aims to achieve longer term.  This design view below provides an overview of proposed components that interact to simulate the detection of animals via a network of sensors.  The Echo Simulator provides a way to generate random sound events (with associated location metadata) for the engine to process. In this way, the simulator is simulating the real world audio sensor array. Simulated Sensor data is translated into 'Detection Events' by the Echo Engine which are stored in the Echo Store ready for the HMI client to connect and visualise.  

When a new Detection Event is added into the Echo Store, the API pushes these events to all connected client front end applications so that detections can be visualised in real time.  In addition to the animal classification result, probability information is provided.  This probably information can be integrated and combined by the HMI to form a 'heatmap' of animal sounds per species over a given area of interest.  

![Data Flow](ProductConceptDesign.jpg) 

#### Project Echo Components

The following sub-sections describe each of the major components of the Project Echo system.  These components.  The Project Echo team members will contribute across these components (as a matrix organisation) to apply their skills in cross dispine virtual teams.

##### Echo Prototype

Echo Prototype is a prototype classification engine and data processing pipeline will demostrate that it is possible to process data from raw labelled sound clips through to classification of animals species.  This prototype engine will be in the form of a series of Jupyter notebook files which must be manually run in sequence (to drive the data pipeline)  It is expected that these prototype notebook files will provide an environment that allows data scientists to further develop and refine models to increase classification performance into the future.

The Echo Prototype will be packages into a python package for commandline deployment.  In this way, the prototype will behave similar to the way the openai 'Whisper' model is deployed - which originally inspired the approach for this project.

##### Echo Engine

The Echo Engine component takes requests from an API in the form of sound clip, executes the classification model and returns the classification result back to the API to distribute to Echo HMI end points.  The classicication model will be trained offline and deployed into the Echo Engine once the performance of the model is validated.

##### Echo API

The Echo API provides a secure end point service for remote applications to process sound clips and return classification results.  The Echo API uses the Echo Engine to do the processing in its behalf.  The API also forwards data to the Echo Store which will persist the sound clips for further subsequent research.

##### Echo HMI (Human Machine Interface)

The Echo HMI provides a front end user experience (UX) for capturing sound clips of animals and allow them to be uploaded to the Echo API service.  The HMI allows interactive interaction with the system.  In the long run, it is expected that the majority of data requests to the Echo API Service will originate from a suite of remote sensors in the field (e.g. across the Otways region).

##### Echo Store

The Echo Store is a modern database solution for storing raw audio files as requested via the Echo API Service.  This store acts a record of the sound sample requests and provides an opportunity for researchers to later label these records and further fine tune the quality of the species classification model.

##### Echo Simulator

The Echo simulator


#### Data Pre-Processing Pipeline 

A diagram of the data flows and major system components identified so far is contained in the  diagram below.  To assist in the interative approach to design, the diagram link is provided to an editable online diagramming tool called 'miro'.  To edit this diagram follow the link provided here using the  password 'projectecho'. https://miro.com/app/board/uXjVPBXmVOA=/

![Data Flow](DataflowOverview.jpg) 

Note: If any updates are made in miro, these will need to be manually exported as an image file and uploaded to github to keep this design material up to date.

The pre-processing pipeline converts audio files into a consistent format ready for processing by the mel-spectrogram pipeline.  In order to ensure that each audio file is treated in a consistent way, audio files will need to be converted to a consistent bitrate, normalised and converted to the same format.  In addition, the pipeline cuts the audio file into 5 second slices which will result in an assocaited mel-spectram for each slice.  All the parameters mentioned on the design diagram above a hyper-parameters of the system that will ultimately need to be tuned end-to-end to ensure compatiblity with the image classification modeling inputs (e.g. changing the HOP parameter will change with width of the resulting spectrogram image).  The initial set of hyperparamters were derived from the Exploratory Data Analysis (EDA) as captured in this repository.

#### Mel-Spectrogram Generation Pipeline

The mel-spectrogram pipeline generates spectrogram images from each audio sample by splitting into small audio clips of equal size and also generating augumentations - such as frequency shifting the audio signal.  The literature indicates that model generalisation performance can be improved through augmenting the audio input samples.  Augmentation can be performed directly on the audio amplitude signal and/or the generated mel-spectrogram images.  One type of augmentation masks some part of the time or frequency space of the spectrogram image randomly so the model doesn't learn any biases towards specific image artefacts in certain frequency bands - e.g. learning to classify animals on the basis of how much background noise is in the audio signal rather than the structure of the animal sound.

#### Classificaton Model Training

The model training uses pre-computed spectrogram images which have been split into training, test and validation subsets.   Pre-computation of the spectrograms is required as the process of performing fast fourier transforms to derive the energy at each frequency band is a relatively expensive and slow computational task.  Since a large number of input samples are likely, it is not possible to leverage caching in the input pipeline.  The initial protoype will leverage pre-trained image classification models to perform the down-stream classification task.  This classification will be a multi-class classification task that will provide a probability distribution across the known species the model is trained on.  

Care must be taken in the implementation to ensure that augmentation is only applied to the training data set.  If augmented files are included in the validation or test set then data leakage is likely to occur and the metrics will not be valid (as augmented signals are not real world samples).