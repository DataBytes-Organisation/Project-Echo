## Bioacoustics-Classification-Tool

### Capstone Project (A) - Bioacoustics Classification Tool

#### Use Case Analysis

<br>

|       | <strong>User Application - Installation</strong> |
| ----------- | ----------- |
| Description   | General user with an android or iphone installs application to their mobile device.  The device could be a phone, tablet or touch screen laptop device.     
| Actors | General User, Researcher |
| Sequence | 1. User searches relevant application store for the app. |
| |2. User installs the application |

<br>

|       | <strong>User Application - First Time Run</strong> |
| ----------- | ----------- |
| Description   | General user with an android or iphone runs the application for the first time and registers using 3rd party authentication options such as google.
| Actors | General User, Researcher |
| Sequence | 1. User runs the app. |
| |2. Splash screen shows while loading. |
| |3. User is required to accept privacy terms and conditions (i.e. all uploaded recordings will be kept for research purposes) |
| |4. User logs in an is presented with microphone recording option. |

<br>

|       | <strong>User Application - Upload Sound Sample</strong> |
| ----------- | ----------- |
| Description   | General user records small 5 second clip of animal sound and uploads it for classification
| Actors | General User, Researcher |
| Sequence | 1. User clicks the record button |
| |2. Screen show indicator that recording is in progress |
| |3. User confirms that recording can be uploaded (in case audio captured something that might breach privacy)|
| |4. Application uploads the sample to the API server and discards local sample |
| |4. API server records sample in database for future model training |
| |5. API server returns with classification result |
| |6. App displays classification result to the user |

<br>

|       | <strong>API Server - Store and Process </strong>  |
| ----------- | ----------- |
| Description   | This describes how the back end API will likely work    
| Actors | Developer|
| Sequence | Note: API is running all the time |
| | Note: Only authenticated requests are accepted |
| | 1. API Recieves request to accept sound clip |
| | 2. API Transfers file and stores in local database server |
| | 3. API Calls python model to perform inference on the sound sample |
| | 4. API Determines if the model confidence is high enough for a response |
| | 5. API Replys back classification response or not classified to original requestor |

<br>


|       | <strong>PIP Package - Execute pre-trained model </strong>  |
| ----------- | ----------- |
| Description   | This describes how a developer can use pip to install our package and execute the model    
| Actors | Developer|
| Sequence | Note: ```pip install git+https://github.com/stephankokkas/Project-Echo/```|
| Sequence | Note: GitHub URL will change once the project is handed off but currently Stephan Kokkas is the owner.|
| | Note: No authentication required |
| | 1. Pip package is installed |
| | 2. Load pre-trained model loaded into memory |
| | 3. Pass model valid audio file through docuemented params |
| | 4. Model will execute in the background and return back classification response or not classified to original requestor |

<br>

|       | <strong>Application - Proof Of Concept Only</strong>  |
| ----------- | ----------- |
| Description   | This describes the proof of concept system use case     
| Actors | Developer|
| Sequence | 1. Developer runs local jupyter notebook |
| | 2. Jupyter notebook visually displays processing stages during the data pipeline |
| | 3. Jupyter notebook prints classification results to the notebook render output |

<br>

##### Front End Application for Otways integration
