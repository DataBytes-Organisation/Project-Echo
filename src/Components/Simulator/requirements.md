# Simulator Requirements

### Functionality
-The simulator shall represent n microphones distributed across the Otways national park.
-The simulator shall semi-randomly generate animal sounds:
>> Nominal animal behaviours are expected (sound producuing frequencies and normal travel paths)

-The simulator shall detect sounds produced by animals through the microphones and link them together:
>> If microphone n has detected a sound, then it should ping all neighbouring node microphones and query if they too have detected a sound
>>> if so, then triangulare origin of audio...

-The simulator shall triangulate the source of the audio file using the signal strength of the audio file from each detecting microphone, as well as the lat and long values of each microphone.
-The simulator shall output the raw audio file (or combination of audio files) from the microphone(s) that detected the sound.
>> Need to define which audio sample is sent to the messaging bus with the triangulated Lat Long pair. 
>>> The simulator shall then send the raw audio file and the corresponding lat and long values to a messaging bus.

### Performance
-The simulator shall process animal sound detection and triangulation in semi-real-time.
>> Simulated real-time
-The simulator shall perform triangulation with a high degree of accuracy, using appropriate algorithms and techniques.
>> Define algorithms...
-The simulator shall handle multiple animal sound detections and triangulations concurrently.

### Reliability
-The simulator shall be able to recover from errors and continue operating.
-The simulator shall have built-in error handling and logging mechanisms.

### Usability
-The simulator shall have a user-friendly interface for configuring and launching the simulation.
>> Who will start the simulator and how?
-The simulator shall have clear documentation on how to use and configure it.
>> Include in Handover doc too
-The simulator shall be easy to set up and run on a virtual machine.
>> Defining environments with fixed library versions in the .yaml file

### Compatibility
-The simulator shall be written in Python.
-The simulator shall be compatible with a predefined messaging bus protocol.

### Security
-The simulator shall implement appropriate security measures to protect sensitive data and prevent unauthorized access.
>> Most likely at a VM level
-The simulator shall be designed with privacy in mind, ensuring that personal data is not exposed or collected.
