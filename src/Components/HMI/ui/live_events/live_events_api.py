from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import os
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
def load_random_event():
    """Load a random live event from the datasets."""
    all_events = []
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                data = pd.read_excel(file_path)
                if 'Animal' in data.columns:
                    # Clean the dataset
                    data = data.fillna("N/A")  # Replace NaN with a placeholder
                    random_row = data.sample(1).to_dict(orient='records')[0]
                    all_events.append(random_row)
                else:
                    print(f"'Animal' column not found in {file_path}")
            else:
                print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")

    if all_events:
        random_event = random.choice(all_events)
        print("Selected event:", random_event)  # Debugging output
        return random_event
    else:
        return {"event": "No events available", "time": "N/A"}



@app.route('/live_events_api', methods=['GET'])
def live_events():
    """Endpoint to fetch a random live event."""
    event = load_random_event()
    return jsonify(event)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)