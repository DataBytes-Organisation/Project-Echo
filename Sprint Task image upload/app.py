import os
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from werkzeug.utils import secure_filename
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Configuration for SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///animals.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Directory to save uploaded files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model for predicting animal in images
image_model = InceptionV3(weights='imagenet')

# Database model for storing image and recognized animal type
class AnimalImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    animal_type = db.Column(db.String(50), nullable=False)

# Initialize the database
with app.app_context():
    db.create_all()

@app.route('/')
def home():
    return "Server is running. Use the /upload route to upload files."

@app.route('/home', methods=['POST',"GET"])
def upload_image():
    return render_template("C:/Users/LENOVO/Desktop/Sprint Task/main.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        print("No image part found in request")
        return jsonify({"error": "No image part"}), 400

    image_file = request.files['image']
    print(f"Image file received: {image_file.filename}")

    if image_file.filename == '':
        print("No file selected")
        return jsonify({"error": "No selected file"}), 400

    # Save the image file securely
    filename = secure_filename(image_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(f"Saving image to {filepath}")
    image_file.save(filepath)

    # Process the image and identify the animal using TensorFlow model
    img = keras_image.load_img(filepath, target_size=(299, 299))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = image_model.predict(img_array)
    labels = decode_predictions(predictions, top=1)[0]
    animal_name = labels[0][1]  # Get the name of the predicted animal
    print(f"Predicted animal: {animal_name}")

    # Save the information to the database
    new_image = AnimalImage(filename=filename, animal_type=animal_name)
    db.session.add(new_image)
    db.session.commit()

    return jsonify({'tags': [animal_name]})


    return jsonify({'tags': [animal_name]})

if __name__ == '__main__':
    app.run(debug=True)
