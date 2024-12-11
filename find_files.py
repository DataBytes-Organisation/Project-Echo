import os

def find_files_and_databases(root_path):
    all_files = []
    database_files = []

    # Walk through all directories and files
    for root, dirs, files in os.walk(root_path):
        for file in files:
            full_path = os.path.join(root, file)
            all_files.append(full_path)
            if file.endswith('.db'):
                database_files.append(full_path)

    return all_files, database_files

# Specify the root directory of Project Echo
project_path = r"C:\Users\<YourUsername>\Documents\Project-Echo"

# Find files and databases
all_files, db_files = find_files_and_databases(project_path)

# Save to text files
with open('all_files.txt', 'w') as all_files_output:
    all_files_output.write('\n'.join(all_files))

with open('database_files.txt', 'w') as db_files_output:
    db_files_output.write('\n'.join(db_files))

print(f"Found {len(all_files)} files in total.")
print(f"Found {len(db_files)} database files.")
print("Results saved to 'all_files.txt' and 'database_files.txt'.")
