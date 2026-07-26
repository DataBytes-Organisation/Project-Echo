#!/usr/bin/env python3
"""
Audio classification accuracy test script.

This module tests the MobileNet bird detector against labelled audio
samples. It provides an AccuracyTester class which can load a TFLite
model, convert audio files to mel spectrograms, run inference and
aggregate accuracy statistics.
"""

import os
import json
import librosa
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import defaultdict, Counter
import time
from tabulate import tabulate
# Load system configuration
try:
    from config.system_config import SC
except Exception:
    SC = {}

class AccuracyTester:
    """
    Accuracy tester for a TFLite MobileNet bird detector.

    The class loads a TFLite model and its class names, converts audio
    files to mel spectrograms compatible with the model and runs
    inference to produce accuracy metrics and a detailed JSON report.
    """
    def __init__(self, model_path, class_names_path, test_audio_dir):
        """
        Initialise the AccuracyTester.

        Args:
            model_path (str): Path to the TFLite model file.
            class_names_path (str): Path to the JSON file containing class
                names in order of model outputs.
            test_audio_dir (str): Directory containing test audio organised
                by species (subdirectories per species).
        """
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.test_audio_dir = test_audio_dir
        self.model = None
        self.class_names = []
        self.results = []
        
    def load_model(self):
        """
        Load the TensorFlow Lite interpreter and class names.

        This method initialises the TFLite interpreter, allocates tensors
        and reads the class names JSON file into memory.
        """
        print("Loading model and class names...")

        # Load the TFLite interpreter
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

        # Get input and output details from the interpreter
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"Model loaded: Input {self.input_details[0]['shape']}, Output {self.output_details[0]['shape']}")

        # Load class names from JSON
        with open(self.class_names_path, 'r') as f:
            self.class_names = json.load(f)
        print(f"Loaded {len(self.class_names)} class names")
        
    def audio_to_mel_spectrogram(self, audio_path):
        """
        Convert an audio file to a mel spectrogram suitable for model input.

        Args:
            audio_path (str): Path to the audio file to convert.

        Returns:
            numpy.ndarray or None: A batch-shaped float32 array with shape
            (1, 224, 224, 3) suitable for the model, or None if an error
            occurred while processing the audio file.

        Description:
            The function loads the audio at 22050 Hz, ensures a 1 second
            length by padding or sampling a random 1 second chunk, computes
            a mel spectrogram with 224 mel bands, converts to dB scale,
            normalises to the range 0-1, resizes to 224x224 and converts
            the single channel to a 3-channel image by replication.
        """
        try:
            # Load audio file and ensure it is mono
            SAMPLE_RATE = int(SC.get('AUDIO_SAMPLE_RATE', 22050))
            CLIP_LENGTH = int(SC.get('AUDIO_CLIP_DURATION', 1))
            CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_LENGTH)

            audio_data, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE, duration=None, mono=True)

            # If audio is shorter than the clip length, pad with silence
            if len(audio_data) < CLIP_SAMPLES:
                audio_data = np.pad(audio_data, (0, CLIP_SAMPLES - len(audio_data)), mode='constant')
            
            # If audio is longer than the clip length, take a random clip-length chunk
            elif len(audio_data) > CLIP_SAMPLES:
                max_start = len(audio_data) - CLIP_SAMPLES
                start_pos = np.random.randint(0, max_start + 1)
                audio_data = audio_data[start_pos:start_pos + CLIP_SAMPLES]
            
            # Generate mel spectrogram
            # Mel spectrogram parameters from config with sensible defaults
            N_MELS = int(SC.get('AUDIO_MELS', 224))
            N_FFT = int(SC.get('AUDIO_NFFT', 2048))
            FMAX = int(SC.get('AUDIO_FMAX', SAMPLE_RATE // 2))

            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio_data, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, fmax=FMAX
            )
            
            # Convert to dB scale
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            
            # Normalise to the range [0, 1]
            mel_spectrogram_normalized = (mel_spectrogram_db - mel_spectrogram_db.min()) / (
                mel_spectrogram_db.max() - mel_spectrogram_db.min() + 1e-8
            )
            
            # Resize to 224x224 using interpolation
            from scipy.ndimage import zoom
            height_ratio = 224 / mel_spectrogram_normalized.shape[0]
            width_ratio = 224 / mel_spectrogram_normalized.shape[1]
            mel_resized = zoom(mel_spectrogram_normalized, (height_ratio, width_ratio), order=1)
            
            # Convert to 3 channels by repeating the single channel
            mel_spectrogram_3ch = np.stack([mel_resized, mel_resized, mel_resized], axis=-1)
            
            # Add batch dimension and ensure correct shape and type
            mel_spectrogram_batch = np.expand_dims(mel_spectrogram_3ch, axis=0).astype(np.float32)
            
            return mel_spectrogram_batch
            
        except Exception as e:
            # Report processing error without special symbols
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def predict(self, spectrogram):
        """
        Run inference on a preprocessed spectrogram.

        Args:
            spectrogram (numpy.ndarray): A batch-shaped spectrogram array, for
                example the output of :meth:`audio_to_mel_spectrogram` with
                shape (1, 224, 224, 3).

        Returns:
            tuple: (predicted_class_index (int) or None,
                    confidence (float),
                    probabilities (numpy.ndarray) or None)

        Description:
            The method sets the input tensor on the TFLite interpreter,
            invokes the interpreter, converts logits to probabilities using
            a numerically stable softmax implementation, and returns the
            predicted class index together with its confidence.
        """
        try:
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], spectrogram)

            # Run inference
            self.interpreter.invoke()

            # Get output logits and convert to probabilities
            logits = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / np.sum(exp_logits)

            # Determine predicted class and confidence
            predicted_class = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class])

            return predicted_class, confidence, probabilities

        except Exception as e:
            # Return None on error and report without special symbols
            print(f"Prediction error: {e}")
            return None, 0.0, None
    
    def get_audio_files(self, directory):
        """
        Retrieve all audio files from a directory recursively.

        Args:
            directory (str or pathlib.Path): Directory to search for audio
                files.

        Returns:
            list[pathlib.Path]: List of Path objects for matching audio files.
        """
        audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}
        audio_files = []
        
        for file_path in Path(directory).rglob('*'):
            if file_path.suffix.lower() in audio_extensions:
                audio_files.append(file_path)
        
        return audio_files
    
    def run_tests(self):
        """
        Run accuracy tests on all audio files in the test directory.

        The method iterates over species subdirectories under the test
        audio directory, converts each audio file to a spectrogram and
        runs the model to collect per-species and overall accuracy
        statistics.
        """
        print("\nTesting audio classification accuracy")
        print("=" * 60)

        # Get all species directories
        species_dirs = [d for d in Path(self.test_audio_dir).iterdir() if d.is_dir()]

        total_files = 0
        correct_predictions = 0
        species_results = defaultdict(lambda: {'total': 0, 'correct': 0, 'predictions': []})

        print(f"Found {len(species_dirs)} species directories")

        for species_dir in species_dirs:
            true_species = species_dir.name
            print(f"\nTesting: {true_species}")

            # Check if species is in the model classes and warn if not
            if true_species not in self.class_names:
                print(f"Warning: '{true_species}' not found in model classes - will check predictions anyway")

            # Get all audio files for this species
            audio_files = self.get_audio_files(species_dir)
            print(f"   Found {len(audio_files)} audio files")

            for audio_file in audio_files:
                print(f"   Processing: {audio_file.name}")

                # Convert to spectrogram suitable for the model
                spectrogram = self.audio_to_mel_spectrogram(str(audio_file))
                if spectrogram is None:
                    continue

                # Make prediction
                predicted_class_idx, confidence, probabilities = self.predict(spectrogram)
                if predicted_class_idx is None:
                    continue

                predicted_species = self.class_names[predicted_class_idx]

                # Record results for later reporting
                is_correct = predicted_species == true_species
                total_files += 1
                if is_correct:
                    correct_predictions += 1

                species_results[true_species]['total'] += 1
                species_results[true_species]['correct'] += is_correct
                species_results[true_species]['predictions'].append({
                    'file': audio_file.name,
                    'predicted': predicted_species,
                    'confidence': confidence,
                    'correct': is_correct
                })

                # Store detailed result for JSON export and later analysis
                self.results.append({
                    'file_path': str(audio_file),
                    'true_species': true_species,
                    'predicted_species': predicted_species,
                    'confidence': confidence,
                    'correct': is_correct
                })

                # Print brief result to the console without special symbols
                status = "Correct" if is_correct else "Incorrect"
                print(f"      {status} - Predicted: {predicted_species} (confidence: {confidence:.3f})")

        return species_results, total_files, correct_predictions
    
    def generate_report(self, species_results, total_files, correct_predictions):
        """
        Generate a detailed accuracy report and print summary statistics.

        Args:
            species_results (dict): Per-species aggregated results produced by
                :meth:`run_tests`.
            total_files (int): Total number of files processed.
            correct_predictions (int): Number of correct predictions.

        Returns:
            dict: A dictionary summarising overall accuracy, total files,
                  correct predictions, per-species results and average
                  confidence.
        """
        print(f"\nACCURACY TEST RESULTS")
        print("=" * 80)

        # Overall accuracy
        overall_accuracy = (correct_predictions / total_files) * 100 if total_files > 0 else 0
        print(f"Overall Accuracy: {correct_predictions}/{total_files} ({overall_accuracy:.2f}%)")

        # Per-species accuracy table
        table_data = []
        for species, data in species_results.items():
            accuracy = (data['correct'] / data['total']) * 100 if data['total'] > 0 else 0
            table_data.append([
                species,
                data['total'],
                data['correct'],
                f"{accuracy:.1f}%"
            ])

        # Sort by accuracy descending
        table_data.sort(key=lambda x: float(x[3].replace('%', '')), reverse=True)

        print(f"\nPer-species results:")
        headers = ["Species", "Total Files", "Correct", "Accuracy"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        # Top predictions analysis
        print(f"\nPrediction analysis:")
        all_predictions = Counter()
        for result in self.results:
            all_predictions[result['predicted_species']] += 1

        print(f"\nMost common predictions:")
        for species, count in all_predictions.most_common(10):
            percentage = (count / total_files) * 100 if total_files > 0 else 0
            print(f"   {species}: {count} files ({percentage:.1f}%)")

        # Confidence distribution
        confidences = [result['confidence'] for result in self.results]
        if confidences:
            avg_confidence = np.mean(confidences)
            print(f"\nConfidence statistics:")
            print(f"   Average confidence: {avg_confidence:.3f}")
            print(f"   Min confidence: {np.min(confidences):.3f}")
            print(f"   Max confidence: {np.max(confidences):.3f}")

            # High confidence predictions
            high_conf_results = [r for r in self.results if r['confidence'] >= 0.7]
            high_conf_accuracy = (sum(1 for r in high_conf_results if r['correct']) / len(high_conf_results) * 100) if high_conf_results else 0
            print(f"   High confidence (>=0.7): {len(high_conf_results)}/{total_files} files ({(len(high_conf_results)/total_files*100) if total_files>0 else 0:.1f}%)")
            print(f"   High confidence accuracy: {high_conf_accuracy:.1f}%")

        # Detailed errors
        errors = [r for r in self.results if not r['correct']]
        if errors:
            print(f"\nClassification errors ({len(errors)} total):")
            for error in errors[:10]:  # Show first 10 errors
                print(f"   {Path(error['file_path']).name}")
                print(f"      True: {error['true_species']}")
                print(f"      Predicted: {error['predicted_species']} (confidence: {error['confidence']:.3f})")

        return {
            'overall_accuracy': overall_accuracy,
            'total_files': total_files,
            'correct_predictions': correct_predictions,
            'species_results': dict(species_results),
            'average_confidence': np.mean(confidences) if confidences else 0
        }
    
    def save_results(self, results_summary):
        """
        Save the test results summary and detailed results to a JSON file.

        Args:
            results_summary (dict): Summary dictionary returned by
                :meth:`generate_report`.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"accuracy_test_results_{timestamp}.json"
        
        # Prepare data for JSON serialisation
        json_data = {
            'timestamp': timestamp,
            'model_path': self.model_path,
            'test_audio_dir': self.test_audio_dir,
            'summary': results_summary,
            'detailed_results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2)

        # Inform the user where results have been saved
        print(f"\nResults saved to: {filename}")

def main():
    """Main function to run accuracy tests"""
    # Configuration
    model_path = "/workspace/Model/Model.tflite"
    class_names_path = "/workspace/Model/class_names.json"
    test_audio_dir = "/workspace/test_audio"
    
    print("MobileNet Bird Detector - Accuracy Test")
    print("=" * 50)
    
    # Check whether files exist
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    if not os.path.exists(class_names_path):
        print(f"Class names file not found: {class_names_path}")
        return
    
    if not os.path.exists(test_audio_dir):
        print(f"Test audio directory not found: {test_audio_dir}")
        return
    
    # Initialise and run tests
    tester = AccuracyTester(model_path, class_names_path, test_audio_dir)
    
    try:
        # Load model
        tester.load_model()
        
        # Run tests
        species_results, total_files, correct_predictions = tester.run_tests()
        
        # Generate report
        results_summary = tester.generate_report(species_results, total_files, correct_predictions)
        
        # Save results
        tester.save_results(results_summary)
        
        print(f"\nTesting completed successfully.")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
