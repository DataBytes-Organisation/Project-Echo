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

3. The simulator shall internall detect sounds produced by animals through the microphones and link them together:
    1. If microphone n has detected a sound, then it should ping all neighbouring node microphones and query if they too have detected a sound
    2. Upon detection, then triangulare origin of audio in order to send the latitude and longitude of the sound
    3. The sensor closest to the vocalistion will be responsible for being the source of the sent audio and will populate with the triangulated latitude and longitude of the animal.

4. The simulator shall triangulate the source of the audio file using the signal strength of the audio file from each detecting microphone, as well as the lat and long values of each microphone.
5. The simulator shall output the raw audio file (or combination of audio files) from the microphone(s) that detected the sound.
    1. Need to define which audio sample is sent to the messaging bus with the triangulated Lat Long pair. 
        1. The simulator shall then send the raw audio file and the corresponding lat and long values to a messaging bus.

### Performance
6. The simulator shall process animal sound detection and triangulation in semi-real-time.
    1. Simulated real-time
7. The simulator shall perform triangulation with a high degree of accuracy, using appropriate algorithms and techniques.
    1. Define algorithms...
8. The simulator shall handle multiple animal sound detections and triangulations concurrently.

### Reliability
9. The simulator shall be able to recover from errors and continue operating.
10. The simulator shall have built-in error handling and logging mechanisms.

### Usability
11. The simulator shall have a user-friendly interface for configuring and launching the simulation.
    1. The simulator shall allow the user to configure the distribution and types of animal species
    2. The simulator shall allow the user to specify a geographical area of interest for simulation.
12. The simulator shall have clear documentation on how to use and configure it.
    1. Include in Handover doc too
13. The simulator shall be easy to set up and run on a virtual machine.
    1. Defining environments with fixed library versions in the .yaml file

### Compatibility
14. The simulator shall be written in Python.
15. The simulator shall be compatible with a predefined messaging bus protocol.

### Security
16. The simulator shall implement appropriate security measures to protect sensitive data and prevent unauthorized access.
    1. Most likely at a VM level
17. The simulator shall be designed with privacy in mind, ensuring that personal data is not exposed or collected.
