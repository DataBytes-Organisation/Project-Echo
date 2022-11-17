## Bioacoustics-Classification-Tool

### Capstone Project (A) - Bioacoustics Classification Tool

#### Use Case Analysis

<br>

|       | Use Case Detail |
| ----------- | ----------- |
| Name      | User Application - Installation       |
| Description   | General user with an android or iphone installs application to their mobile device.  The device could be a phone, tablet or touch screen laptop device.     
| Actors | General User, Researcher |
| Sequence | 1. User searches relevant application store for the app. |
| |2. User installs the application |

<br>

|       | Use Case Detail |
| ----------- | ----------- |
| Name      | User Application - First Time Run       |
| Description   | General user with an android or iphone runs the application for the first time and registers using 3rd party authentication options such as google.
| Actors | General User, Researcher |
| Sequence | 1. User runs the app. |
| |2. Splash screen shows while loading. |
| |3. User is required to accept privacy terms and conditions (i.e. all uploaded recordings will be kept for research purposes) |
| |4. User logs in an is presented with microphone recording option. |

<br>

|       | Use Case Detail |
| ----------- | ----------- |
| Name      | User Application - Upload Sound Sample      |
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

|       | Use Case Detail |
| ----------- | ----------- |
| Name      | Application - Proof Of Concept Only       |
| Description   | This describes the proof of concept system use case     
| Actors | Developer|
| Sequence | 1. Developer runs local jupyter notebook |
| | 2. Jupyter notebook visually displays processing stages during the data pipeline |
| | 3. Jupyter notebook prints classification results to the notebook render output |

<br>

##### Front End Application for Otways integration
