from flask import Flask, request, jsonify, send_from_directory, make_response
import os
import hashlib
import json
import tensorflow as tf
import tensorflowjs as tfjs
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'json', 'h5', 'model.json'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_md5(data):
    md5_hash = hashlib.md5()
    md5_hash.update(data.encode('utf-8'))
    return md5_hash.hexdigest()

def log_request(f):
    def decorated_function(*args, **kwargs):
        print(f"Request: {request.url}, Data: {request.form}")
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

@app.route('/upload/spectrogram', methods=['POST'])
@log_request
def upload_spectrogram():
    if 'project_id' not in request.form or 'spectrogram' not in request.form or 'labels' not in request.form:
        return jsonify({'error': 'Missing project_id, labels, or spectrogram'}), 400

    project_id = request.form['project_id']
    labels = request.form['labels']
    spectrogram = request.form['spectrogram']

    print(f"Received labels: {labels}")
    print(f"Received spectrogram: {spectrogram}")

    try:
        labels_json = json.loads(labels)
        spectrogram_json = json.loads(spectrogram)
    except json.JSONDecodeError as e:
        return jsonify({'error': f'Invalid JSON format: {str(e)}'}), 400

    md5_hash = calculate_md5(spectrogram)
    filename = f"{md5_hash}.json"

    project_folder = os.path.join(app.config['UPLOAD_FOLDER'], project_id, 'jsondata')
    os.makedirs(project_folder, exist_ok=True)

    filepath = os.path.join(project_folder, filename)
    
    data = {
        'labels': labels_json,
        'spectrogram': spectrogram_json,
        'md5': md5_hash
    }

    with open(filepath, 'w') as f:
        json.dump(data, f)

    return jsonify({'message': 'File saved', 'md5': md5_hash}), 200

@app.route('/upload/aiModel', methods=['POST'])
@log_request
def upload_ai_model():
    project_id = request.form.get('project_id')
    file = request.files.get('file')

    if not project_id or not file:
        return jsonify({'error': 'Missing project_id or file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    filename = secure_filename(file.filename)
    extension = filename.rsplit('.', 1)[1].lower()
    project_folder = os.path.join(app.config['UPLOAD_FOLDER'], project_id, 'aiModels')
    os.makedirs(project_folder, exist_ok=True)
    filepath = os.path.join(project_folder, filename)

    file.save(filepath)

    if extension == 'json':
        # Convert TensorFlow.js model to TensorFlow model
        tf_model_path = filepath.replace('.json', '.h5')
        tfjs.converters.save_keras_model(filepath, tf_model_path)
        return jsonify({'message': 'TensorFlow.js model converted and saved', 'path': tf_model_path}), 200

    return jsonify({'message': 'File saved', 'path': filepath}), 200

@app.route('/files/<project_id>', methods=['GET'])
def list_files(project_id):
    project_folder = os.path.join(app.config['UPLOAD_FOLDER'], project_id)
    files = {}
    for folder in ['jsondata', 'aiModels']:
        folder_path = os.path.join(project_folder, folder)
        if os.path.exists(folder_path):
            files[folder] = os.listdir(folder_path)
    return jsonify(files), 200

@app.route('/files/view/<project_id>/<folder>/<filename>', methods=['GET'])
def view_file(project_id, folder, filename):
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], project_id, folder)
    if os.path.exists(folder_path):
        response = make_response(send_from_directory(folder_path, filename))
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, proxy-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    return jsonify({'error': 'File not found'}), 404

@app.route('/models/<project_id>/<model_name>', methods=['POST'])
@log_request
def run_inference(project_id, model_name):
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], project_id, 'aiModels', model_name)
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model not found'}), 404

    model = tf.keras.models.load_model(model_path)
    data = request.json.get('data')

    if not data:
        return jsonify({'error': 'No data provided for inference'}), 400

    predictions = model.predict(data)
    return jsonify({'predictions': predictions.tolist()}), 200

@app.route('/test', methods=['POST'])
@log_request
def test_endpoint():
    data = request.get_json()
    response = {
        'received': True,
        'data': data
    }
    return jsonify(response), 200

@app.route('/')
def phealthcheck():
    return "Hello from Python engine backend!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=50999, debug=True)

