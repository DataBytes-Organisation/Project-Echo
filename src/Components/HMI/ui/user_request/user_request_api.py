from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

requests_file = "../Data/user_requests.csv"
if not os.path.exists(requests_file):
    pd.DataFrame(columns=["username", "email", "animal", "request_type", "details", "timestamp"]).to_csv(requests_file, index=False)

@app.route('/submit_request', methods=['POST'])
def submit_request():
    """API endpoint to handle user requests."""
    try:
        # Get JSON data from the request
        data = request.get_json()
        print(data)  # Debugging output to check incoming request data
        username = data.get("username")
        email = data.get("email")
        animal = data.get("animal")
        request_type = data.get("request_type")
        details = data.get("details")
        timestamp = pd.Timestamp.now().isoformat()

        # Validate the required fields
        if not (username and email and request_type):
            return jsonify({"success": False, "error": "Missing required fields"}), 400

        # Save the request to a CSV file
        new_request = pd.DataFrame([{
            "username": username,
            "email": email,
            "animal": animal,
            "request_type": request_type,
            "details": details,
            "timestamp": timestamp
        }])
        new_request.to_csv(requests_file, mode='a', header=False, index=False)

        return jsonify({"success": True, "message": "Request submitted successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)