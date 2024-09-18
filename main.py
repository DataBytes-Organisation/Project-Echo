import numpy as np
import os
import pickle
import librosa

from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

# Configuration for SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///animals.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Directory to save uploaded files
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model for predicting animals in images
image_model = InceptionV3(weights='imagenet')

#Setting the label encoders for sound prediction
l2 = open("static/labels.pkl","rb")
label = pickle.load(l2)
label_encoder = LabelEncoder()
label_encoder.fit_transform(label)

# Database model for storing image filenames and recognized animal types
class AnimalImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    animal_type = db.Column(db.String(50), nullable=False)

# Initialize the database
with app.app_context():
    #db.drop_all() #uncomment this to clear the database
    db.create_all()

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"error": "No image part in the request"}), 400

        image_file = request.files['image']

        if image_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save the image file securely
        filename = secure_filename(image_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(filepath)

        # Process the image and identify the animal using TensorFlow model
        img = keras_image.load_img(filepath, target_size=(299, 299))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = image_model.predict(img_array)
        labels = decode_predictions(predictions, top=1)[0]
        animal_name = labels[0][1]  # Get the name of the predicted animal

        # Save the information to the database
        new_image = AnimalImage(filename=filename, animal_type=animal_name)
        db.session.add(new_image)

        try:
            db.session.commit()
            print(f"Successfully saved {filename} with label {animal_name} to the database.")
        except Exception as e:
            db.session.rollback()
            print(f"Failed to save {filename} to the database: {e}")
            return jsonify({"error": f"Failed to save the image: {str(e)}"}), 500

        # Return the image URL and tags (animal name)
        image_url = url_for('static', filename=f'uploads/{filename}', _external=True)
        return jsonify({
            'image_url': image_url,
            'tags': [animal_name]
        })

    return render_template('upload.html')

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if request.method == 'POST':
        if 'audio' not in request.files:
            return jsonify({"error": "No audio part in the request"}), 400

        audio_file = request.files['audio']

        if audio_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save the audio file securely
        filename = secure_filename(audio_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(filepath)
        try:
            predicted_animal = predict_class(filepath)
            return jsonify({
            'animal_type': predicted_animal
            })
        except Exception as e:
            print("Error has occured")
            return jsonify({"error": "Invalid request method"}), 405


@app.route('/gallery')
def gallery():
    images = AnimalImage.query.all()
    return render_template('gallery.html', images=images)


# Function to print all records from the AnimalImage table
def print_all_records():
    images = AnimalImage.query.all()
    for image in images:
        print(f"ID: {image.id}, Filename: {image.filename}, Animal Type: {image.animal_type}")

#Functions that predicts the animal from audio
def feature_extractor_from_array(y, sr):
    print("feature_extractor")
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    max_length = 500
    if mfccs.shape[1] > max_length:
        mfccs = mfccs[:, :max_length]
    else:
        padding = max_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, padding)), mode='constant')
    return mfccs

# Prediction function
def predict_class(audio_file):
    print("Inside predict class")
    loaded_model = load_model('static/animal_only_model.h5')
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = feature_extractor_from_array(y, sr)
    mfccs = np.expand_dims(mfccs, axis=(0, -1))  # Add batch and channel dimensions
    prediction = loaded_model.predict(mfccs)
    predicted_class = np.argmax(prediction, axis=1)
    return label_encoder.inverse_transform(predicted_class)[0]


if __name__ == '__main__':
    app.run(debug=True)
