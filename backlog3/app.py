from flask import Flask, render_template, request, jsonify
import math

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/projected-movement', methods=['POST'])
def projected_movement():
    # Get region data from frontend
    data = request.json
    x, y, movement_factor, time = data['x'], data['y'], data['movement_factor'], data['time']

    # Calculate projected movement (simple linear example)
    new_x = x + movement_factor * time * math.cos(math.radians(45))  # 45 degrees as an example angle
    new_y = y + movement_factor * time * math.sin(math.radians(45))

    # Return the projected coordinates
    return jsonify({'new_x': round(new_x, 2), 'new_y': round(new_y, 2)})

if __name__ == '__main__':
    app.run(debug=True)
