import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import librosa
import importlib.util
from _model_config import MODEL_NAME

# Define the directory containing the model files and the test data directory
model_dir = r'C:\Users\pkmaz\Desktop\Project-Echo\src\Prototypes\engine\Transfer Learning Models'
test_data_dir = r'C:\Users\pkmaz\Downloads\Test dataset\Test dataset'

# Function to dynamically load models based on their names
def load_model(model_name, model_dir):
    model_path = os.path.join(model_dir, f'{model_name}.py')
    
    # Load the model definition from the .py file
    spec = importlib.util.spec_from_file_location(model_name, model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Assume the model creation function is named 'create_model' and returns a compiled model
    model = module.create_model()
    
    # Load pretrained weights
    weights_path = os.path.join(model_dir, 'models', model_name, 'variables', 'variables')
    model.load_weights(weights_path)
    
    return model

# Function to load and preprocess audio data
def load_audio_files(data_dir, sample_rate=22050, n_mels=128, max_pad_len=862):
    file_paths = list(pathlib.Path(data_dir).glob('**/*.wav'))
    data = []
    labels = []
    for file_path in file_paths:
        audio, sr = librosa.load(file_path, sr=sample_rate)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        if log_mel_spec.shape[1] > max_pad_len:
            log_mel_spec = log_mel_spec[:, :max_pad_len]
        else:
            pad_width = max_pad_len - log_mel_spec.shape[1]
            log_mel_spec = np.pad(log_mel_spec, pad_width=((0, 0), (0, pad_width)), mode='constant')
        data.append(log_mel_spec)
        labels.append(file_path.parent.name)
    return np.array(data), np.array(labels)

# Function to evaluate and compare models
def compare_models(model, test_data, test_labels):
    results = {}
    predictions = model.predict(test_data)
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(test_labels, axis=1))
    loss = model.evaluate(test_data, test_labels, verbose=0)[0]
    results[model.name] = {'loss': loss, 'accuracy': accuracy}
    return results

# Function to plot model comparison
def plot_comparison(results):
    model_names = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in model_names]
    losses = [results[model]['loss'] for model in model_names]
    
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy', color='tab:blue')
    ax1.bar(model_names, accuracies, color='tab:blue', alpha=0.6, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='tab:red')
    ax2.plot(model_names, losses, color='tab:red', marker='o', linestyle='dashed', linewidth=2, label='Loss')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.title('Model Comparison')
    plt.show()

# Load the model
model = load_model(MODEL_NAME, model_dir)

# Load and preprocess test data
test_data, test_labels = load_audio_files(test_data_dir)

# Convert labels to one-hot encoding
label_set = sorted(set(test_labels))
label_to_index = {label: index for index, label in enumerate(label_set)}
test_labels = np.array([label_to_index[label] for label in test_labels])
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=len(label_set))

# Expand dimensions to match the expected input shape for the models
test_data = np.expand_dims(test_data, axis=-1)

# Compare model
results = compare_models(model, test_data, test_labels)

# Plot comparison
plot_comparison(results)
