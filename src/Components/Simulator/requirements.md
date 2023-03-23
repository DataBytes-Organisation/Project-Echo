# Simulator Requirements

### Functionality
1. The simulator shall represent n microphones distributed across the Otways national park.

2. The simulator shall represent m animals distributed around the area of interest.

3. The simulator shall simulate animal movement along a brownian motion path from its initial position.

4. The simulator shall semi-randomly generate animal sounds:
    1. Nominal animal behaviours are expected (sound producuing frequencies and normal travel paths)
    2. The animal sounds will be randomly selected from the Echo Store for the assigned animal species
    3. The animal sounds will be a randomly selected 5 second clip of the assigned animal species.
    4. The frequency of animal sound vocalisation shall be configurable and associated with each species 

5. The simulator shall internall detect sounds produced by animals through the microphones and link them together:
    1. If microphone n has detected a sound, then it should ping all neighbouring node microphones and query if they too have detected a sound
    2. Upon detection, then triangularise origin of audio in order to send the latitude and longitude of the sound
    3. The sensor closest to the vocalistion will be responsible for being the source of the sent audio and will populate with the triangulated latitude and longitude of the animal.

6. The simulator shall triangulate the source of the audio file using time of arrival information from each detecting microphone, as well as the lat and long values of each microphone.

5. The simulator shall output the raw audio file (or combination of audio files) from the microphone(s) that detected the sound.
    1. Audio will be sent in accordance with the AnimalSound message on the API and message bus interfaces.
    2. The message will be populated with meta-data in accordance with the interface including time stamp, latitude, longitude, sensor id and other information.

### Performance
6. The simulator shall process animal sound detection and triangulation in semi-real-time.
    1. Simulated real-time
    2. The simulator shall allow the user to speed up or slow the simulation by a supplied speed factor.

7. The simulator shall handle multiple animal sound detections and triangulations concurrently.

### Reliability
9. The simulator shall be able to recover from errors and continue operating.

10. The simulator shall have built-in error handling and logging mechanisms with sufficient detail to enable debugging and integration of the Echo system

### Usability
11. The simulator shall have a user-friendly interface for configuring and launching the simulation.
    1. The simulator shall allow the user to configure the distribution and types of animal species
    2. The simulator shall allow the user to specify a geographical area of interest for simulation.
    3. The simulator shall show the historical track paths of each animal

12. The simulator shall support test modes including periodically generating an audio message to assist with integration of interfaces.

13. The simulator shall have clear documentation on how to use and configure it to support handover and longevity of project Echo including:
    1. A summary of simulator functionality in the Handover documentation.
    2. Detailed documentation of the source code
    3. A small tutorial with an end-to-end walkthrough from installation to running in cloud.

14. The simulator shall be capably of being deployed to the google cloud platform GCP and run on a virtual machine.
    1. The simulator shall have a well defined software environment with fixed library versions in the .yaml file

### Compatibility
14. The simulator shall be written in Python.

15. The simulator shall be compatible with a predefined messaging bus protocol.

### Security
16. The simulator shall implement appropriate security measures to protect sensitive data and prevent unauthorized access.

17. The simulator shall be designed with privacy in mind, ensuring that personal data is not exposed or collected.
