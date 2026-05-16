import pandas as pd

# Paths to the Excel files
excel_files = [
    r"C:\Users\ragul\OneDrive\Documents\Project-Echo\src\Components\HMI\AI\bio_master_A.xlsx",
    r"C:\Users\ragul\OneDrive\Documents\Project-Echo\src\Components\HMI\AI\bio_master_B.xlsx",
    r"C:\Users\ragul\OneDrive\Documents\Project-Echo\src\Components\HMI\AI\bio_master_C.xlsx",
    r"C:\Users\ragul\OneDrive\Documents\Project-Echo\src\Components\HMI\AI\bio_master_D.xlsx",
    r"C:\Users\ragul\OneDrive\Documents\Project-Echo\src\Components\HMI\AI\bio_master_E.xlsx"
]

# Initialize empty list to hold DataFrames
all_data_list = []

# Read each Excel file and append to the list
for file in excel_files:
    df = pd.read_excel(file)
    all_data_list.append(df)

# Concatenate all DataFrames into one
all_data = pd.concat(all_data_list, ignore_index=True)

# Export to CSV
all_data.to_csv('animals_data.csv', index=False)

# Export to JSON
all_data.to_json('animals_data.json', orient='records', lines=True)

print("Data exported successfully!")

