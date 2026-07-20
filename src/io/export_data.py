import pandas as pd
import os

# Paths to the Excel files in the shared Data folder
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
base_data_dir = os.path.join(root_dir, 'Data', 'Components', 'HMI', 'AI')
excel_files = [
    os.path.join(base_data_dir, 'bio_master_A.xlsx'),
    os.path.join(base_data_dir, 'bio_master_B.xlsx'),
    os.path.join(base_data_dir, 'bio_master_C.xlsx'),
    os.path.join(base_data_dir, 'bio_master_D.xlsx'),
    os.path.join(base_data_dir, 'bio_master_E.xlsx')
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

