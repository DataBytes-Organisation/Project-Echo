import pandas as pd
import json
import os

# Paths to the Excel files for animal data
animal_data_files = [
    r"C:\Users\ragul\OneDrive\Documents\Project-Echo\src\Components\HMI\AI\bio_master_A.xlsx",
    r"C:\Users\ragul\OneDrive\Documents\Project-Echo\src\Components\HMI\AI\bio_master_B.xlsx",
    r"C:\Users\ragul\OneDrive\Documents\Project-Echo\src\Components\HMI\AI\bio_master_C.xlsx",
    r"C:\Users\ragul\OneDrive\Documents\Project-Echo\src\Components\HMI\AI\bio_master_D.xlsx",
    r"C:\Users\ragul\OneDrive\Documents\Project-Echo\src\Components\HMI\AI\bio_master_E.xlsx"
]

# Initialize an empty DataFrame to hold all animal data
all_animal_data = pd.DataFrame()

# Read each Excel file and append the data
for file in animal_data_files:
    if os.path.exists(file):
        df = pd.read_excel(file)
        all_animal_data = pd.concat([all_animal_data, df], ignore_index=True)
    else:
        print(f"File not found: {file}")

# Print columns of animal data
print("Animal Data Columns:", all_animal_data.columns)

# Load the movement data from JSON
movements_file = r"C:\Users\ragul\OneDrive\Documents\Project-Echo\src\Components\MongoDB\init\movements.json"
if os.path.exists(movements_file):
    with open(movements_file, 'r') as f:
        movements_data = json.load(f)
else:
    print(f"File not found: {movements_file}")
    movements_data = []

# Convert movement data to a DataFrame
movements_df = pd.DataFrame(movements_data)

# Print columns of movement data
print("Movement Data Columns:", movements_df.columns)

# Select relevant columns
all_animal_data = all_animal_data[["Animal", "Common Name", "JSON"]]
movements_df = movements_df[["species", "timestamp", "animalId", "animalTrueLLA"]]

# Combine animal data and movement data
combined_data = pd.merge(all_animal_data, movements_df, left_on="Animal", right_on="species", how="left")

# Export combined data to CSV
combined_data.to_csv('combined_animal_data.csv', index=False)
print("Combined data exported to combined_animal_data.csv")

# Export combined data to JSON
combined_data.to_json('combined_animal_data.json', orient='records')
print("Combined data exported to combined_animal_data.json")
