import json

# Path to the original data file (movements.json)
original_data_file = r"C:\Users\ragul\OneDrive\Documents\Project-Echo\src\Components\MongoDB\init\movements.json"

# Load the original data from the JSON file
with open(original_data_file, 'r') as f:
    original_data = json.load(f)

# Clean data by removing the altitude (third value in the 'animalTrueLLA' list)
cleaned_data = []
for record in original_data:
    # Ensure 'animalTrueLLA' contains valid coordinates
    if isinstance(record["animalTrueLLA"], list) and len(record["animalTrueLLA"]) == 3:
        # Only keep the lat and lon, remove the altitude (third value)
        cleaned_coordinates = record["animalTrueLLA"][:2]  # Exclude the altitude (third value)
        cleaned_record = record.copy()
        cleaned_record["animalTrueLLA"] = cleaned_coordinates
        cleaned_data.append(cleaned_record)
    else:
        print(f"Invalid coordinate data: {record['animalTrueLLA']}")

# Save the cleaned data to a new JSON file
cleaned_data_file = r"C:\Users\ragul\OneDrive\Documents\Project-Echo\src\Components\MongoDB\init\cleaned_animal_movements.json"
with open(cleaned_data_file, 'w') as f:
    json.dump(cleaned_data, f, indent=4)

print(f"Cleaned data saved to {cleaned_data_file}")
 
