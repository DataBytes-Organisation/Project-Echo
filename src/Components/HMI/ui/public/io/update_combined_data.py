import json
import requests

# Load JSON file
with open('src/io/combined_animal_data.json', 'r', encoding='utf-8') as f:
    animals = json.load(f)

# Get microphone coordinates from FastAPI
response = requests.get("http://localhost:9000/hmi/microphones")
microphones = response.json()

# Create mapping: microphone_id => [lat, lon]
mic_coords_map = {}
for mic in microphones:
    try:
        mic_id = int(mic["_id"]) if isinstance(mic["_id"], int) else int(mic.get("microphone_id", 0))
        mic_coords_map[mic_id] = mic["microphoneLLA"][:2]
    except Exception as e:
        print(f"Skipped microphone data: {mic}")

# Add coordinates to animal JSON
for animal in animals:
    mic_id = animal.get("microphone_id")
    coords = mic_coords_map.get(mic_id)
    animal["microphone_coords"] = coords if coords else None

with open('src/io/combined_animal_data.json', 'w', encoding='utf-8') as f:
    json.dump(animals, f, indent=2)

print("Microphone coordinates have been added.")
