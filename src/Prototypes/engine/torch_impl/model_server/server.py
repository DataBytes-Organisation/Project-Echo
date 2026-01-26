from flask import Flask, request, jsonify
import numpy as np
from inference_engine import ModelLoader
import os

app = Flask(__name__)

# CONFIGURATION
# Set this to "onnx" or "tflite" in Docker environment variables
BACKEND = os.environ.get("INFERENCE_BACKEND", "tflite")

# Select default file based on backend
if BACKEND == "onnx":
    DEFAULT_MODEL_FILE = "model.onnx"
else:
    DEFAULT_MODEL_FILE = "model.tflite"

MODEL_PATH = os.environ.get("MODEL_FILE", DEFAULT_MODEL_FILE)

# Initialize Engine
print(f"Starting Model Server with {BACKEND} using {MODEL_PATH}...")
try:
    engine = ModelLoader.get_engine(BACKEND, MODEL_PATH)
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model: {e}")
    engine = None

@app.route('/v1/models/echo_model/versions/1:predict', methods=['POST'])
def predict():
    if engine is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    try:
        # 1. Parse JSON (Matches original Engine format)
        data = request.json
        # 'inputs' comes as a list from the engine
        input_list = data.get("inputs")
        
        # 2. Convert to Numpy
        input_array = np.array(input_list, dtype=np.float32)
        
        # 3. Predict
        result = engine.predict(input_array)
        
        # 4. Return in TF Serving format
        return jsonify({"outputs": result.tolist()})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8501)