from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Sample data for regions
REGION_DATA = {
    "Savannah": {"sightings": 150, "vegetation_density": "Low", "movement_patterns": "Stable"},
    "Rainforest": {"sightings": 230, "vegetation_density": "High", "movement_patterns": "Dynamic"},
    "Desert": {"sightings": 45, "vegetation_density": "Sparse", "movement_patterns": "Minimal"},
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/region-data', methods=['GET'])
def get_region_data():
    return jsonify(REGION_DATA)

if __name__ == '__main__':
    app.run(debug=True)
