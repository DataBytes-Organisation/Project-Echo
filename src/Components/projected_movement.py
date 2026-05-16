import json
import matplotlib.pyplot as plt

# Load movement data (update the file path if needed)
with open(r"C:\Users\ragul\OneDrive\Documents\Project-Echo\src\Data\Animals\animal_movements.json") as f:
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
