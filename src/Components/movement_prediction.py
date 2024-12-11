import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime

# Load the movement data from the file
file_path = r"C:\Users\ragul\OneDrive\Documents\Project-Echo\src\Components\MongoDB\init\movements.json"
with open(file_path, 'r') as f:
    movements_data = json.load(f)

# Function to clean and filter the coordinates (latitude, longitude)
def clean_coordinates(coords):
    """
    Ensure that only valid lat/lon data is used (ignore extra values like altitude).
    :param coords: List of coordinates (latitude, longitude, altitude)
    :return: Cleaned coordinates as [latitude, longitude] or None if invalid
    """
    if isinstance(coords, list) and len(coords) >= 2:
        # Only take the first two values (latitude, longitude)
        return coords[:2]
    else:
        return None  # Return None if data is invalid

# Function to predict future coordinates using linear regression
def predict_future_movement(coords, steps=3):
    """
    Predict future coordinates using polynomial regression.
    :param coords: List of [lat, lon] pairs
    :param steps: Number of future points to predict
    :return: List of predicted coordinates
    """
    if len(coords) < 2:
        return []

    # Ensure coords is a list of pairs [lat, lon]
    coords = np.array([clean_coordinates(coord) for coord in coords if clean_coordinates(coord) is not None])

    # If no valid data after cleaning, return an empty list
    if coords.shape[0] < 2:
        return []

    latitudes = coords[:, 0]
    longitudes = coords[:, 1]

    # Use index as a time variable
    time = np.arange(len(coords)).reshape(-1, 1)

    # Polynomial feature transformation (degree 2)
    poly = PolynomialFeatures(degree=2)
    time_poly = poly.fit_transform(time)

    # Linear regression for latitude and longitude prediction
    model_lat = LinearRegression().fit(time_poly, latitudes)
    model_lon = LinearRegression().fit(time_poly, longitudes)

    # Generate future time steps
    future_time = np.arange(len(coords), len(coords) + steps).reshape(-1, 1)
    future_time_poly = poly.transform(future_time)

    # Predict future coordinates
    future_lats = model_lat.predict(future_time_poly)
    future_lons = model_lon.predict(future_time_poly)

    # Combine latitudes and longitudes into a list of coordinates
    return list(zip(future_lats, future_lons))

# Plot the movement of each animal
plt.figure(figsize=(10, 6))

for entry in movements_data:
    species = entry.get("species", "Unknown")
    coords = entry.get("animalTrueLLA", [])
    
    # Skip invalid data
    if not coords or len(coords) < 2:
        print(f"Skipping invalid data for {species}")
        continue
    
    # Clean the coordinates (latitude, longitude only)
    coords = [clean_coordinates(coord) for coord in coords if clean_coordinates(coord) is not None]

    # If there are valid coordinates, proceed with prediction and plotting
    if coords:
        # Predict future movements based on historical data
        predicted_coords = predict_future_movement(coords, steps=3)
        
        # Plot the historical data
        coords = np.array(coords)
        plt.plot(coords[:, 1], coords[:, 0], marker="o", label=f"{species} (Observed)")

        # Plot the predicted future movements
        if predicted_coords:
            predicted_coords = np.array(predicted_coords)
            plt.plot(predicted_coords[:, 1], predicted_coords[:, 0], marker="x", linestyle="--", label=f"{species} (Predicted)")

# Customize the plot
plt.title("Projected and Predicted Animal Movements")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc='upper right')
plt.grid(True)

# Save and show the plot
plt.savefig("movement_prediction_with_future.png")
plt.show()
