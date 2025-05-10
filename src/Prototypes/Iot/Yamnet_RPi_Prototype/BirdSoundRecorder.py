"""
Bird Sound Recorder
This program detects specific animal/bird sounds using YAMNet and saves audio clips
capturing 5 seconds before and after detection.
"""

import numpy as np
import pyaudio
import wave
import time
import os
import datetime
import collections
import threading
from yamnetOnPi import (
    FORMAT, CHANNELS, SAMPLE_RATE, CHUNK_SIZE,
    initialize_audio, load_model, load_labels, open_audio_stream,
    process_audio, get_predictions, cleanup
)

# Target classes to detect with their confidence thresholds
TARGET_CLASSES = {
    "Animal": 0.6,
    "Wild animals": 0.6,
    "Bird": 0.6,
    "Bird vocalization, bird call, bird song": 0.6,
    "Chirp, tweet": 0.6
}

# Calculate buffer size for 5 seconds of audio
# Each chunk is approximately CHUNK_SIZE/SAMPLE_RATE seconds long
SECONDS_PER_CHUNK = CHUNK_SIZE / SAMPLE_RATE
BUFFER_CHUNKS = int(5 / SECONDS_PER_CHUNK) + 1  # Add 1 to ensure we have at least 5 seconds
AFTER_CHUNKS = int(5 / SECONDS_PER_CHUNK) + 1  # Chunks to record after detection

def create_timestamp():
    """Create a timestamp string for filenames"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def save_audio(audio_data, filename):
    """Save audio data to a WAV file"""
    # Ensure recordings directory exists
    os.makedirs("recordings", exist_ok=True)
    
    # Convert list of audio chunks to single byte array
    audio_bytes = b''.join(audio_data)
    
    # Create the WAV file
    filepath = os.path.join("recordings", filename)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 2 bytes for FORMAT=paInt16
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_bytes)
    
    print(f"Audio saved to {filepath}")
    return filepath

def record_after_detection(stream, num_chunks):
    """Record additional chunks after detection"""
    chunks = []
    for _ in range(num_chunks):
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        chunks.append(data)
    return chunks

def main():
    """Main function to detect sounds and save audio clips"""
    print("Starting Bird Sound Recorder...")
    
    # Initialise audio

    audio = initialize_audio()
    if not audio:
        return
    
    # Load model and labels
    interpreter = load_model()
    if not interpreter:
        audio.terminate()
        return
    
    labels = load_labels()
    if not labels:
        audio.terminate()
        return
    
    # Open audio stream
    stream = open_audio_stream(audio)
    if not stream:
        audio.terminate()
        return
    
    # Create a circular buffer to store recent audio
    audio_buffer = collections.deque(maxlen=BUFFER_CHUNKS)
    
    # Flags for recording management
    currently_recording = False
    last_detection_time = 0
    cooldown_period = 10  # Seconds between recordings
    
    print(f"Listening for sounds... Buffer size: {BUFFER_CHUNKS} chunks ({BUFFER_CHUNKS * SECONDS_PER_CHUNK:.1f} seconds)")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            print(f"Processing at: {time.time()}")
            # Read audio data
            audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            
            # Add to the circular buffer
            audio_buffer.append(audio_data)
            
            # Skip prediction if in cooldown period
            current_time = time.time()
            if currently_recording or (current_time - last_detection_time) < cooldown_period:
                continue
            
            # Process audio for prediction
            audio_np = process_audio(audio_data)
            
            # Get predictions
            predictions = get_predictions(interpreter, audio_np, labels)
            
            # Check if any target classes are detected
            detected = False
            detected_class = None
            detected_score = 0
            
            for label, score in predictions:
                if label in TARGET_CLASSES and score >= TARGET_CLASSES[label]:
                    detected = True
                    detected_class = label
                    detected_score = score
                    break
            
            if detected:
                print(f"\n🔊 Detected {detected_class} with confidence {detected_score:.3f}")
                currently_recording = True
                last_detection_time = current_time
                
                # Create a copy of the buffer (contains 5 seconds before detection)
                buffer_copy = list(audio_buffer)
                
                # Define recording handler function
                def handle_recording():
                    nonlocal currently_recording
                    print(f"Recording {AFTER_CHUNKS * SECONDS_PER_CHUNK:.1f} more seconds...")
                    
                    # Record after detection
                    after_chunks = record_after_detection(stream, AFTER_CHUNKS)
                    
                    # Combine before and after chunks
                    all_chunks = buffer_copy + after_chunks
                    
                    # Create filename and save
                    timestamp = create_timestamp()
                    filename = f"{detected_class.replace(' ', '_').replace(':', '')}_{timestamp}.wav"
                    save_audio(all_chunks, filename)
                    
                    currently_recording = False
                    print("Ready to detect again...")
                
                # Start recording in a separate thread
                recording_thread = threading.Thread(target=handle_recording)
                recording_thread.start()
    
    except KeyboardInterrupt:
        print("\nStopping Bird Sound Recorder...")
    finally:
        # Clean up resources
        cleanup(stream, audio)

if __name__ == "__main__":
    main()