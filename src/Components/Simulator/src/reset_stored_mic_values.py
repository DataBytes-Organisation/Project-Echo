import json
import os
from entities.entity import Entity

# Load JSON data from file
_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'mics_info.json')
with open(_path, 'r') as f:
    mics_info = json.load(f)

# Update the latitude and longitude values
for mic in mics_info:
    e = Entity()
    new_lat, new_lon, new_alt = e.randLatLong()
    mic['latitude'] = new_lat
    mic['longitude'] = new_lon
    mic['elevation'] = new_alt

# Write the updated data back to the JSON file
with open(_path, 'w') as f:
    json.dump(mics_info, f, indent=4)
