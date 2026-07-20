import json
import matplotlib.pyplot as plt
import os

# Load movement data from the shared Data folder
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
file_path = os.path.join(root_dir, 'Data', 'Animals', 'animal_movements.json')

with open(file_path, 'r') as f:
    data = json.load(f)

# Plot movement data for each animal
for animal in data:
    coords = animal["movement"]
    x, y = zip(*coords)  # Unzips the movement coordinates into x and y
    plt.plot(x, y, label=animal["name"], marker="o")  # Plot movement

plt.title("Projected Animal Movements")
plt.xlabel("X Coordinates")
plt.ylabel("Y Coordinates")
plt.legend()  # Add a legend to the plot
plt.savefig("projected_movement.png")  # Save the plot as an image
plt.show()  # Display the plot
