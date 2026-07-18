import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the cleaned movement data from the new file
cleaned_data_file = r"C:\Users\ragul\OneDrive\Documents\Project-Echo\src\Components\MongoDB\init\cleaned_animal_movements.json"

with open(cleaned_data_file, 'r') as f:
    cleaned_data = json.load(f)

# Function to predict future movement based on cleaned coordinates
def predict_future_movement(coords, steps=3):
    if len(coords) < 2:
        print("Insufficient data for prediction.")
        return []

    # Use Linear Regression to predict future movement
    model = LinearRegression()

    # Prepare data for regression (treat latitudes as X and longitudes as Y)
    latitudes = np.array([coord[0] for coord in coords]).reshape(-1, 1)
    longitudes = np.array([coord[1] for coord in coords])

    model.fit(latitudes, longitudes)

    # Predict future coordinates
    last_lat = latitudes[-1]
    predicted_coords = []
    for i in range(steps):
        predicted_long = model.predict([[last_lat + i]])  # Incrementing the latitude step by step
        predicted_coords.append([last_lat + i, predicted_long[0]])

    return predicted_coords

# Loop through the data and make predictions for each species
for record in cleaned_data:
    species = record['species']
    coords = record['animalTrueLLA']
    
    print(f"Species: {species}")
    print(f"Raw animalTrueLLA: {coords}")

    # Ensure the coordinates are valid
    if isinstance(coords, list) and len(coords) == 2:
        # Cleaned Coordinates (Lat, Lon)
        cleaned_coords = [coords]  # We treat each as a single point for prediction
        print(f"Cleaned Coordinates: {cleaned_coords}")
        
        # Make predictions for the next 3 steps
        predicted_coords = predict_future_movement(cleaned_coords, steps=3)
        print(f"Predicted Coordinates: {predicted_coords}")
        
        # Plot the predicted and observed data
        lats, lons = zip(*cleaned_coords)  # Unpacking cleaned coordinates (Lat, Lon)
        plt.plot(lons, lats, marker="o", label=f"{species} (Observed)")

        if predicted_coords:
            pred_lats, pred_lons = zip(*predicted_coords)
            plt.plot(pred_lons, pred_lats, marker="x", label=f"{species} (Predicted)")

    else:
        print(f"Invalid coordinate data for {species}. Skipping prediction.")

# Customize plot
plt.title("Projected Animal Movements")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc='upper right')

# Save the plot
plt.savefig("projected_movement.png")
plt.show()

