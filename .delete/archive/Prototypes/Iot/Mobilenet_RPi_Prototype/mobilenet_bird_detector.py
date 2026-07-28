"""
MobileNet Bird Sound Detector
This programme takes in audio, processes it into 1-second chunks, converts to mel spectrograms,
and feeds them into a MobileNet model for bird detection. On positive detection, it saves the audio file.
"""

import numpy as np
import pyaudio
import wave
import time
import os
import datetime
import collections
import threading
import librosa
import json
from scipy import signal
from pathlib import Path

# Load system configuration (SC) if available
try:
    from config.system_config import SC
except Exception:
    # Minimal fallback config if system_config not available
    SC = {}

# Use TensorFlow Lite for performance
try:
    import tflite_runtime.interpreter as tflite
    print("Using TensorFlow Lite Runtime")
except ImportError:
    import tensorflow.lite as tflite
    print("Using TensorFlow Lite from TensorFlow")

# Audio parameters (sourced from config.system_config.SC when present)
FORMAT = pyaudio.paInt16
CHANNELS = 1
# Prefer SC values, fall back to sensible defaults used previously
SAMPLE_RATE = int(SC.get('AUDIO_SAMPLE_RATE', 22050))
# CHUNK_SIZE corresponds to 1 second of audio at SAMPLE_RATE
CHUNK_SIZE = int(SC.get('CHUNK_SIZE', SAMPLE_RATE))
CHUNK_DURATION = float(SC.get('CHUNK_DURATION', 1.0))

# Mel spectrogram parameters - sourced from config when possible
# Note: keep defaults compatible with older embedded values
N_MELS = int(SC.get('AUDIO_MELS', 224))
N_FFT = int(SC.get('AUDIO_NFFT', 2048))
HOP_LENGTH = None  # Will be calculated dynamically based on model width
FMIN = int(SC.get('AUDIO_FMIN', 0))
FMAX = int(SC.get('AUDIO_FMAX', SAMPLE_RATE // 2))
TOP_DB = int(SC.get('AUDIO_TOP_DB', 80))

# Model parameters - sourced from config when possible
MODEL_INPUT_SHAPE = (
    int(SC.get('MODEL_INPUT_IMAGE_HEIGHT', 224)),
    int(SC.get('MODEL_INPUT_IMAGE_WIDTH', 224)),
    int(SC.get('MODEL_INPUT_IMAGE_CHANNELS', 3)),
)

# Detection parameters
DETECTION_THRESHOLD = 0.7  # Confidence threshold for positive detection
BUFFER_SIZE = 1  # Number of chunks to keep in buffer (1 second total)
COOLDOWN_PERIOD = 0  # Seconds between recordings

class MobileNetBirdDetector:
    """
    Main class for MobileNet-based bird sound detection from audio input.
    
    Attributes:
        model_path (str): Path to the TensorFlow Lite model file
        class_names_path (str): Path to the JSON file containing class names
        audio_device_index (int): Index of audio input device to use
    """
    
    def __init__(self, model_path="Model/Model.tflite", 
                 class_names_path="Model/class_names.json",
                 audio_device_index=None,
                 log_all_predictions=False):
        """
        Initialise the bird detector with specified model and audio settings.
        
        Args:
            model_path (str): Path to TensorFlow Lite model file
            class_names_path (str): Path to JSON file with class names
            audio_device_index (int, optional): Audio device index, None for default
            log_all_predictions (bool): If True, log top 5 predictions every second
        """
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.audio_device_index = audio_device_index  # None = default device
        self.log_all_predictions = log_all_predictions
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.class_names = None
        self.audio = None
        self.stream = None
        self.audio_buffer = collections.deque(maxlen=BUFFER_SIZE)
        self.currently_recording = False
        self.last_detection_time = 0
        
    def load_model(self):
        """
        Load the TensorFlow Lite model from file.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            print(f"Loading TFLite model from {self.model_path}...")
            
            # Check if the file exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load TFLite model
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print("TFLite model loaded successfully!")
            print(f"Input shape: {self.input_details[0]['shape']}")
            print(f"Output shape: {self.output_details[0]['shape']}")
            return True
            
        except Exception as e:
            print(f"Error loading TFLite model: {e}")
            return False
    
    def load_class_names(self):
        """
        Load class names from JSON file.
        
        Returns:
            bool: True if class names loaded successfully, False otherwise
        """
        try:
            with open(self.class_names_path, 'r') as f:
                self.class_names = json.load(f)
            print(f"Loaded {len(self.class_names)} class names")
            return True
        except Exception as e:
            print(f"Error loading class names: {e}")
            return False
    
    def list_audio_devices(self):
        """
        List all available audio input devices.
        
        Returns:
            list: List of tuples containing (device_index, device_info)
        """
        try:
            if self.audio is None:
                temp_audio = pyaudio.PyAudio()
            else:
                temp_audio = self.audio
                
            print("Available audio input devices:")
            print("-" * 50)
            
            input_devices = []
            for i in range(temp_audio.get_device_count()):
                device_info = temp_audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    input_devices.append((i, device_info))
                    print(f"Device {i}: {device_info['name']}")
                    print(f"  Channels: {device_info['maxInputChannels']}")
                    print(f"  Sample Rate: {device_info['defaultSampleRate']}")
                    print(f"  Host API: {temp_audio.get_host_api_info_by_index(device_info['hostApi'])['name']}")
                    print()
            
            if not input_devices:
                print("No audio input devices found!")
                
            if self.audio is None:
                temp_audio.terminate()
                
            return input_devices
            
        except Exception as e:
            print(f"Error listing audio devices: {e}")
            return []

    def initialise_audio(self):
        """
        Initialise PyAudio for audio input.
        
        Returns:
            bool: True if audio initialised successfully, False otherwise
        """
        try:
            self.audio = pyaudio.PyAudio()
            return True
        except Exception as e:
            print(f"Error initialising audio: {e}")
            return False
    
    def open_audio_stream(self):
        """
        Open audio stream for continuous recording.
        
        Returns:
            bool: True if stream opened successfully, False otherwise
        """
        try:
            # Create stream parameters
            stream_params = {
                'format': FORMAT,
                'channels': CHANNELS,
                'rate': SAMPLE_RATE,
                'input': True,
                'frames_per_buffer': CHUNK_SIZE
            }
            
            # Add input device index if specified
            if self.audio_device_index is not None:
                stream_params['input_device_index'] = self.audio_device_index
                print(f"Using audio input device: {self.audio_device_index}")
            else:
                print("Using default audio input device")
            
            self.stream = self.audio.open(**stream_params)
            return True
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            return False
    
    def audio_to_mel_spectrogram(self, audio_data):
        """
        Convert raw audio data to mel spectrogram following training pipeline.
        
        Args:
            audio_data (bytes): Raw audio data from microphone
            
        Returns:
            numpy.ndarray or None: Processed mel spectrogram with shape (224, 224, 3) or None on error
        """
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            audio_np = audio_np / 32768.0  # Normalise to [-1, 1]
            
            # Ensure we have exactly 1 second of audio
            if len(audio_np) != SAMPLE_RATE:
                if len(audio_np) < SAMPLE_RATE:
                    # Pad with zeros if too short
                    audio_np = np.pad(audio_np, (0, SAMPLE_RATE - len(audio_np)))
                else:
                    # Truncate if too long
                    audio_np = audio_np[:SAMPLE_RATE]
            
            # Following the training pipeline exactly:
            # Calculate hop_length to get desired width (224)
            expected_width = MODEL_INPUT_SHAPE[1]  # 224
            expected_height = MODEL_INPUT_SHAPE[0]  # 224
            
            # Calculate hop_length to achieve target width
            # librosa formula: width = (len(audio) - n_fft) // hop_length + 1
            # We want width = expected_width, so:
            # expected_width = (len(audio) - n_fft) // hop_length + 1
            # hop_length = (len(audio) - n_fft) // (expected_width - 1)
            target_frames = (len(audio_np) - N_FFT) // (expected_width - 1)
            hop_length = max(1, target_frames)
            
            # Compute mel spectrogram with training parameters
            mel_spec = librosa.feature.melspectrogram(
                y=audio_np,
                sr=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=hop_length,
                n_mels=expected_height,  # match model height
                fmin=FMIN,
                fmax=FMAX,
                win_length=N_FFT // 2  # Following training config
            )
            
            # Convert to dB scale (following training pipeline)
            mel_spec_db = librosa.power_to_db(mel_spec, top_db=TOP_DB, ref=1.0)
            
            # Transpose to match training pipeline (swap axis)
            mel_spec_db = np.moveaxis(mel_spec_db, 1, 0)  # Now shape is (width, height)
            
            # Expand dims to add channel dimension: (width, height) -> (width, height, 1)
            image = np.expand_dims(mel_spec_db, -1)
            
            # Repeat to get 3 channels: (width, height, 1) -> (width, height, 3)
            image = np.repeat(image, 3, axis=2)
            
            # Resize to exact model input shape using high-quality interpolation
            from scipy.ndimage import zoom
            height_zoom = MODEL_INPUT_SHAPE[0] / image.shape[0]
            width_zoom = MODEL_INPUT_SHAPE[1] / image.shape[1]
            image = zoom(image, (height_zoom, width_zoom, 1), order=3)  # Cubic interpolation
            
            # Normalise to [0,1] range (following training pipeline)
            image = image - np.min(image)
            image = image / (np.max(image) + 1e-7)  # Add epsilon to avoid division by zero
            
            return image
            
        except Exception as e:
            print(f"Error creating mel spectrogram: {e}")
            return None
    
    def predict_bird(self, mel_spectrogram):
        """
        Make prediction using the TensorFlow Lite model.
        
        Args:
            mel_spectrogram (numpy.ndarray): Preprocessed mel spectrogram
            
        Returns:
            tuple: (predicted_class_name, confidence_score, top_5_predictions) or (None, 0.0, []) on error
                   where top_5_predictions is a list of (class_name, score) tuples.
        """
        try:
            # Prepare input data
            input_data = np.expand_dims(mel_spectrogram, axis=0).astype(np.float32)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            logits = output_data[0]  # Remove batch dimension
            
            # Convert logits to probabilities using softmax
            exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
            probabilities = exp_logits / np.sum(exp_logits)
            
            # Get the top prediction
            predicted_class_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_class_idx]
            predicted_class = self.class_names[predicted_class_idx]

            # Get top 5 predictions
            top_5_indices = np.argsort(probabilities)[-5:][::-1]
            top_5_predictions = [(self.class_names[i], probabilities[i]) for i in top_5_indices]
            
            return predicted_class, confidence, top_5_predictions
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None, 0.0, []
    
    def is_target_bird(self, predicted_class):
        """
        Check if the predicted class should trigger recording - accepts any valid classification.
        
        Args:
            predicted_class (str): The predicted class name from the model
            
        Returns:
            bool: True if the class is valid and should trigger recording, False otherwise
        """
        # Return True for any valid prediction (any species in our model)
        return predicted_class is not None and predicted_class in self.class_names
    
    def create_timestamp(self):
        """
        Create a timestamp string for filenames.
        
        Returns:
            str: Formatted timestamp string (YYYYMMDD_HHMMSS)
        """
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_audio(self, audio_data, predicted_class, confidence):
        """
        Save audio data to a WAV file with species and confidence information.
        
        Args:
            audio_data (list or bytes): Audio data chunks or single byte array
            predicted_class (str): Name of predicted species
            confidence (float): Confidence score from model
            
        Returns:
            str or None: Path to saved file or None on error
        """
        try:
            # Ensure recordings directory exists
            os.makedirs("recordings", exist_ok=True)
            
            # Convert list of audio chunks to single byte array
            if isinstance(audio_data, list):
                audio_bytes = b''.join(audio_data)
            else:
                audio_bytes = audio_data
            
            # Create filename
            timestamp = self.create_timestamp()
            safe_class_name = predicted_class.replace(' ', '_').replace(':', '').replace('/', '_')
            filename = f"{safe_class_name}_{confidence:.3f}_{timestamp}.wav"
            filepath = os.path.join("recordings", filename)
            
            # Create the WAV file
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # 2 bytes for FORMAT=paInt16
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_bytes)
            
            print(f"Audio saved: {filename}")
            return filepath
            
        except Exception as e:
            print(f"Error saving audio: {e}")
            return None
    
    def record_additional_chunks(self, num_chunks):
        """
        Record additional audio chunks after detection for extended context.
        
        Args:
            num_chunks (int): Number of additional chunks to record
            
        Returns:
            list: List of audio data chunks
        """
        chunks = []
        try:
            for _ in range(num_chunks):
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                chunks.append(data)
        except Exception as e:
            print(f"Error recording additional chunks: {e}")
        return chunks
    
    def run_detection(self):
        """
        Main detection loop that continuously processes audio and detects bird sounds.
        
        This method runs indefinitely until interrupted, processing 1-second audio chunks
        and triggering recordings when bird sounds are detected above the confidence threshold.
        """
        print("Starting MobileNet Bird Sound Detector...")
        print(f"Recording ANY species classification from {len(self.class_names)} total classes")
        print(f"Detection threshold: {DETECTION_THRESHOLD}")
        print("Processing 1-second audio chunks")
        print("Listening for any wildlife... Press Ctrl+C to stop")
        
        try:
            while True:
                current_time = time.time()
                
                # Read audio chunk (1 second)
                audio_data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                
                # Add to buffer
                self.audio_buffer.append(audio_data)
                
                # Skip if in cooldown period or currently recording
                if (self.currently_recording or 
                    (current_time - self.last_detection_time) < COOLDOWN_PERIOD):
                    continue
                
                # Convert to mel spectrogram
                mel_spec = self.audio_to_mel_spectrogram(audio_data)
                if mel_spec is None:
                    continue
                
                # Make prediction
                predicted_class, confidence, top_5 = self.predict_bird(mel_spec)

                # Log top 5 predictions if enabled
                if self.log_all_predictions:
                    print(f"Top 5 predictions at {time.strftime('%H:%M:%S')}:")
                    for i, (species, score) in enumerate(top_5):
                        print(f"  {i+1}. {species}: {score:.3f}")
                
                # Check for detection
                if (predicted_class and confidence >= DETECTION_THRESHOLD and 
                    self.is_target_bird(predicted_class)):
                    
                    print(f"\nSPECIES DETECTED: {predicted_class}")
                    print(f"Confidence: {confidence:.3f}")
                    
                    self.currently_recording = True
                    self.last_detection_time = current_time
                    
                    # Create a copy of the buffer (contains last 5 seconds)
                    buffer_copy = list(self.audio_buffer)
                    
                    # Handle recording in separate thread
                    def handle_recording():
                        try:
                            # Record 2 more seconds after detection
                            additional_chunks = self.record_additional_chunks(2)
                            
                            # Combine buffer and additional chunks
                            all_chunks = buffer_copy + additional_chunks
                            
                            # Save the audio
                            self.save_audio(all_chunks, predicted_class, confidence)
                            
                        except Exception as e:
                            print(f"Error in recording thread: {e}")
                        finally:
                            self.currently_recording = False
                            print("Ready for next detection...")
                    
                    # Start recording thread
                    recording_thread = threading.Thread(target=handle_recording)
                    recording_thread.start()
                
                else:
                    # Print periodic status (every 10 seconds)
                    if int(current_time) % 10 == 0:
                        print(f"Listening... Last prediction: {predicted_class or 'None'} ({confidence:.3f})")
                
        except KeyboardInterrupt:
            print("\nStopping Bird Sound Detector...")
        except Exception as e:
            print(f"Error in detection loop: {e}")
    
    def cleanup(self):
        """
        Clean up audio resources and close streams.
        """
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
        print("Cleanup completed")
    
    def start(self):
        """
        Initialise and start the bird detector system.
        
        Returns:
            bool: True if detector started successfully, False otherwise
        """
        try:
            # Initialise components
            if not self.initialise_audio():
                return False
            
            if not self.load_model():
                self.cleanup()
                return False
            
            if not self.load_class_names():
                self.cleanup()
                return False
            
            if not self.open_audio_stream():
                self.cleanup()
                return False
            
            # Start detection
            self.run_detection()
            
        except Exception as e:
            print(f"Error starting detector: {e}")
            return False
        finally:
            self.cleanup()
        
        return True

def main():
    """
    Main function to parse command line arguments and start the bird detector.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='MobileNet Bird Sound Detector')
    parser.add_argument('--device', '-d', type=int, default=None,
                       help='Audio input device index (use --list-devices to see available devices)')
    parser.add_argument('--list-devices', action='store_true',
                       help='List available audio input devices and exit')
    parser.add_argument('--log-all', action='store_true',
                       help='Log top 5 predictions every second')
    parser.add_argument('--model', '-m', default="Model/Model.tflite",
                       help='Path to the TFLite model file')
    parser.add_argument('--classes', '-c', default="Model/class_names.json",
                       help='Path to the class names JSON file')
    
    args = parser.parse_args()
    
    # If list devices requested, do that and exit
    if args.list_devices:
        detector = MobileNetBirdDetector()
        detector.initialise_audio()
        detector.list_audio_devices()
        detector.cleanup()
        return
    
    # Create detector with specified device
    detector = MobileNetBirdDetector(
        model_path=args.model,
        class_names_path=args.classes,
        audio_device_index=args.device,
        log_all_predictions=args.log_all
    )
    
    detector.start()

if __name__ == "__main__":
    main()
