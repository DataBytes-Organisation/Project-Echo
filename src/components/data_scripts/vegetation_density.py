import json
import matplotlib.pyplot as plt

# Load vegetation data (update the file path if needed)
with open(r"C:\Users\ragul\OneDrive\Documents\Project-Echo\src\Components\vegetation_density.json") as f:
    data = json.load(f)

regions = [item["region"] for item in data]
densities = [item["density"] for item in data]

# Create a bar chart
plt.bar(regions, densities, color=['green' if d > 60 else 'yellow' for d in densities])
plt.title('Vegetation Density by Region')
plt.xlabel('Regions')
plt.ylabel('Density (%)')
plt.savefig('vegetation_density.png')
plt.show()
 
