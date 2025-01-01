from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import os
from werkzeug.utils import secure_filename
import random
# Initialize Flask app
app = Flask(__name__)
CORS(app)
# File paths for datasets and user requests
file_paths = [
    r"../Data/bio_master_A.xlsx",
    r"../Data/bio_master_B.xlsx",
    r"../Data/bio_master_C.xlsx",
    r"../Data/bio_master_D.xlsx",
    r"../Data/bio_master_E.xlsx",
]
def load_time_lapse_data(start_time, stop_time):
    """Load animal movement data within a time range."""
    events = []
    try:
        for file_path in file_paths:
            if os.path.exists(file_path):
                data = pd.read_excel(file_path)
                if 'timestamp' in data.columns and 'latitude' in data.columns and 'longitude' in data.columns:
                    # Filter by time range
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                    filtered = data[(data['timestamp'] >= start_time) & (data['timestamp'] <= stop_time)]
                    events.extend(filtered.to_dict(orient='records'))
                else:
                    print(f"Missing required columns in {file_path}")
            else:
                print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error processing file: {e}")
    return events

@app.route('/time_lapse_api', methods=['GET'])
def time_lapse():
    """API endpoint for time lapse data."""
    start_time = request.args.get('start')
    stop_time = request.args.get('stop')

    if not start_time or not stop_time:
        return jsonify({"success": False, "error": "Missing 'start' or 'stop' parameters"}), 400

    try:
        events = load_time_lapse_data(pd.to_datetime(start_time), pd.to_datetime(stop_time))
        return jsonify({"success": True, "events": events})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)