## Simulator Requirements

### Functionality

1. The simulator shall represent n microphones distributed across the Otways National Park.

2. The simulator shall represent m animals distributed within the specified area of interest.

3. The simulator shall simulate animal movement following a Brownian motion path from its initial position.

4. The simulator shall semi-randomly generate animal sounds with the following attributes:

    - Adherence to nominal animal behaviors, such as sound-producing frequencies and typical travel paths.
    - Random selection of animal sounds from the Echo Store, corresponding to the assigned species.
    - Use of a random 5-second sound clip for each assigned species.
    - Configurable frequency of sound vocalization associated with each species.

5. The simulator shall internally detect animal sounds through the microphones, triangulate their origin, and send the latitude and longitude coordinates of the sound source.

6. Detection of a sound at one microphone should prompt pinging of neighboring microphones.

7. The microphone closest to the sound source shall send the audio file and the triangulated coordinates.

8. The simulator shall output the raw audio file(s) from the detecting microphone(s), along with associated metadata, in accordance with the AnimalSound message on the API and message bus interfaces.

### Performance

9. The simulator shall process animal sound detection and triangulation in semi-real-time, with user-configurable simulation speed.

10. The simulator shall concurrently handle multiple animal sound detections and triangulations.

### Reliability

11. The simulator shall feature error recovery capabilities, enabling continued operation after errors.

12. The simulator shall include built-in error handling and logging mechanisms, with sufficient detail for debugging and Echo system integration.

### Usability

13. The simulator shall provide a user-friendly interface for configuration and simulation launch, allowing users to:

    - Configure animal species distribution and types.
    - Specify a geographical area of interest.
    - View historical animal track paths.

14. The simulator shall support test modes, including periodic audio message generation for interface integration assistance.

15. The simulator shall offer comprehensive documentation, including:
    - A summary of simulator functionality in the handover documentation.
Detailed source code documentation.
    - An end-to-end tutorial covering installation and cloud-based operation.

16. The simulator shall support deployment on the Google Cloud Platform (GCP) and operation on a virtual machine, with a well-defined software environment and fixed library versions specified in the .yaml file.

### Compatibility

17. The simulator shall be written in Python.

18. The simulator shall be compatible with a predefined messaging bus protocol.

### Security

19. The simulator shall implement security measures to protect sensitive data and prevent unauthorized access.

20. The simulator shall prioritize privacy, ensuring that personal data is neither exposed nor collected.