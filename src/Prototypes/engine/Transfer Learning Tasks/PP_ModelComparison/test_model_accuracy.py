import tensorflow as tf
import numpy as np
import librosa

# Currently placeholder just to map out the framework for the final implementation (hoping to finish this next semester as a team lead)

# Load the trained models
model_effnet = tf.keras.models.load_model('path_to_saved_model/EfficientNetV2B0.h5')
model_inception = tf.keras.models.load_model('path_to_saved_model/InceptionV3.h5')
model_mobilenetv2 = tf.keras.models.load_model('path_to_saved_model/MobileNetV2.h5')
model_mobilenetv3large = tf.keras.models.load_model('path_to_saved_model/MobileNetV3Large.h5')
model_mobilenetv3small = tf.keras.models.load_model('path_to_saved_model/MobileNetV3Small.h5')
model_resnet = tf.keras.models.load_model('path_to_saved_model/ResNet50V2.h5')


def preprocess_wav(file_path, target_size=(224, 224)):
    # Load the .wav file
    y, sr = librosa.load(file_path, sr=None)
    
    # Convert to mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Resize the spectrogram to the target size
    resized_spectrogram = tf.image.resize(mel_spectrogram_db, target_size)
    
    # Expand dimensions to match the model input (batch_size, height, width, channels)
    input_data = np.expand_dims(resized_spectrogram, axis=-1)  # Add channel dimension
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
    
    return input_data


def make_prediction(model, input_data):
    prediction = model.predict(input_data)
    return np.argmax(prediction, axis=1)  # Get the predicted class

# Example usage
file_path = 'path_to_your_wav_file.wav'
input_data = preprocess_wav(file_path)

# Predict with each model
pred_effnet = make_prediction(model_effnet, input_data)
pred_inception = make_prediction(model_inception, input_data)
pred_mobilenetv2 = make_prediction(model_mobilenetv2, input_data)
pred_mobilenetv3large = make_prediction(model_mobilenetv3large, input_data)
pred_mobilenetv3small = make_prediction(model_mobilenetv3small, input_data)
pred_resnet = make_prediction(model_resnet, input_data)

print("EfficientNetV2B0 Prediction:", pred_effnet)
print("InceptionV3 Prediction:", pred_inception)
print("MobileNetV2 Prediction:", pred_mobilenetv2)
print("MobileNetV3Large Prediction:", pred_mobilenetv3large)
print("MobileNetV3Small Prediction:", pred_mobilenetv3small)
print("ResNet50V2 Prediction:", pred_resnet)

class_labels = ['class1', 'class2', 'class3', ...]  # Replace with your actual class labels

def interpret_prediction(prediction, class_labels):
    return class_labels[prediction[0]]

print("EfficientNetV2B0 Prediction:", interpret_prediction(pred_effnet, class_labels))
print("InceptionV3 Prediction:", interpret_prediction(pred_inception, class_labels))
print("MobileNetV2 Prediction:", interpret_prediction(pred_mobilenetv2, class_labels))
print("MobileNetV3Large Prediction:", interpret_prediction(pred_mobilenetv3large, class_labels))
print("MobileNetV3Small Prediction:", interpret_prediction(pred_mobilenetv3small, class_labels))
print("ResNet50V2 Prediction:", interpret_prediction(pred_resnet, class_labels))

