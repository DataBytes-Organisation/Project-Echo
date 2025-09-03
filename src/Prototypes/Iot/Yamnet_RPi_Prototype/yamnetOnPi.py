# yamnetonPi.py

""" YAMNet Raspberry Pi Audio Classification Script
This script captures audio from a microphone, processes it using the YAMNet model, and outputs the top 5 predictions."""

import numpy as np
import pyaudio
import time
import resampy
import tflite_runtime.interpreter as tflite
import sys
import os
import csv
from utils import create_class_map

# Audio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK_SIZE = 15600  # This matches YAMNet's expected input size


def initialize_audio():
    """Initialize the PyAudio instance."""
    return pyaudio.PyAudio()


def load_model(model_path="yamnet_model/yamnet.tflite"):

    """Load the TFLite model and return the interpreter."""

    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def load_labels(labels_path="yamnet_model/yamnet_labels.txt"):
    """Load and process the labels for the model."""

    try:
        with open(labels_path, "r") as f:
            labels_raw = [line.strip() for line in f.readlines()]
        
        # Get the class map
        class_map = create_class_map()
        
        # Map labels to human-readable names
        labels = []
        for label in labels_raw:

            # The label might be just an ID or might contain other info
            label_id = label.split(',')[0].strip() if ',' in label else label

            #  Get the human-readable name from the class map
            human_name = class_map.get(label_id, label_id)

            # Append to labels list
            labels.append(human_name)
        
        return labels
    
    except Exception as e:
        print(f"Error loading labels: {e}")
        return None


def check_microphone(audio):
    """Check if a microphone is available and return True if found."""
    info = audio.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    
    if num_devices <= 0:
        return False
    
    # Check each device to see if it's an input device
    for i in range(num_devices):

        # Get device info
        device_info = audio.get_device_info_by_index(i)

        # Check if the device has input channels
        if device_info.get('maxInputChannels') > 0:
            print(f"Microphone found: {device_info.get('name')}")
            return True
    
    return False


def open_audio_stream(audio):
    """Open and return an audio stream."""
    try:
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        return stream
    
    except Exception as e:
        print(f"Error opening audio stream: {e}")
        return None


def process_audio(audio_data):
    """Process raw audio data into the format expected by the model."""
    # Convert to numpy array
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    
    # Normalize to -1 to 1
    audio_np = audio_np / 32768.0
    
    return audio_np


def get_predictions(interpreter, audio_np, labels, top_k=5):
    """Run inference and return the top k predictions."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Debug info
    # print(f"Audio shape: {audio_np.shape}")
    # print(f"Model expects: {input_details[0]['shape']}")
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], audio_np)
    
    # Run inference
    interpreter.invoke()
    
    # Get scores
    scores = interpreter.get_tensor(output_details[0]['index']).squeeze()
    
    # Get top k predictions
    top_indices = np.argsort(scores)[-top_k:][::-1]
    
    results = []
    for i in top_indices:
        results.append((labels[i], scores[i]))
    
    return results


def prediction_loop(stream, interpreter, labels):
    """Main loop for audio capture and inference."""
    try:
        while True:
            # Track processing time
            start_time = time.time()
            
            # Read audio data
            audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            
            # Process audio
            audio_np = process_audio(audio_data)
            
            # Get predictions
            predictions = get_predictions(interpreter, audio_np, labels)
            
            # Display results
            print("\nTop 5 predictions:")
            for label, score in predictions:
                print(f"  {label}: {score:.3f}")
                
            print(f"Computation time: {time.time() - start_time:.3f} seconds")
            time.sleep(0.5)  # Wait before processing next chunk
            
    except KeyboardInterrupt:
        print("Stopping...")
        return


def cleanup(stream, audio):
    """Clean up resources."""
    if stream:
        stream.stop_stream()
        stream.close()
    if audio:
        audio.terminate()


def main():
    """Main function to run the YAMNet audio classification."""

    # Initialise audio
    audio = initialize_audio()
    if not audio:
        return
    
    # Check for microphone
    if not check_microphone(audio):
        print("Error: No microphone found. Please connect a microphone and try again.")
        audio.terminate()
        return
    
    # Load model
    interpreter = load_model()
    if not interpreter:
        audio.terminate()
        return
    
    # Load labels
    labels = load_labels()
    if not labels:
        audio.terminate()
        return
    
    # Open audio stream
    stream = open_audio_stream(audio)
    if not stream:
        audio.terminate()
        return
    
    print("Microphone connected. Listening... Press Ctrl+C to stop")
    
    try:
        # Run the prediction loop
        prediction_loop(stream, interpreter, labels)

    finally:
        # Clean up resources
        cleanup(stream, audio)


if __name__ == "__main__":
    main()