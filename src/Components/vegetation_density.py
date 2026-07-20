import json
import matplotlib.pyplot as plt
import os

# Load vegetation data from the shared Data folder
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
with open(os.path.join(root_dir, 'Data', 'Components', 'vegetation_density.json'), 'r') as f:
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
 
