from flask import Flask, request, jsonify, render_template
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image
import os
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
image_model = InceptionV3(weights='imagenet')

# Directory for storing uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Region Mapping
REGION_MAPPING = {
    'lion': 'Savannah',
    'penguin': 'Antarctica',
    'kangaroo': 'Australia',
    'elephant': 'Africa',
    'bear': 'North America',
}

# In-memory storage for images and their tags
animal_data = []

@app.route('/')
def home():
    return render_template('home.html') 

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(filepath)

    # Predict the animal using the pre-trained model
    img = keras_image.load_img(filepath, target_size=(299, 299))
    img_array = preprocess_input(np.expand_dims(keras_image.img_to_array(img), axis=0))
    predictions = image_model.predict(img_array)
    predicted_label = decode_predictions(predictions, top=1)[0][0][1]  # Get the predicted animal

    # Map the animal to its region
    region = REGION_MAPPING.get(predicted_label.lower(), 'Unknown')

    # Store the result in memory
    animal_data.append({"filename": image_file.filename, "animal": predicted_label, "region": region})

    return jsonify({"animal": predicted_label, "region": region})

@app.route('/gallery', methods=['GET'])
def gallery():
    # Display uploaded images with their regions
    return render_template('gallery.html', animal_data=animal_data)

if __name__ == '__main__':
    app.run(debug=True)
