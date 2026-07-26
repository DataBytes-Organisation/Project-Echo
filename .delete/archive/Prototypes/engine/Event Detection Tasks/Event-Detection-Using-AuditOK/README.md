## Event detection using auditOK

This event detection method uses the auditok to finish the task. It will load an audio from the given filepath and segment the audio into clips and saved them to an array.

**How to use event.py**

To use this script, you need to import this script first like this:
```python
    from event import enhanced_event_detection
```    
Then you need to call the functions and receive its return value:
```python
    events = enhanced_event_detection(audio_filepath, SC['AUDIO_SAMPLE_RATE'], energy_threshold)
```    
And extract the segments from the array so that you can pass them for further classification using a loop: 

```python
    for start_sample, end_sample in events:
        segment = waveform[start_sample:end_sample]
```    



    

