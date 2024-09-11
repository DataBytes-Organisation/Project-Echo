import os
import shutil

# Define the mount point inside the container
mount_point = "/mnt/recordings"

# Verify the mount point
if not os.path.isdir(mount_point):
    raise Exception(f"Mount point {mount_point} is not a directory or does not exist")

print(f"Cleaning up files within the volume at {mount_point}")

def delete_directories(directories):
    for dir_path in directories:
        for root, dirs, files in os.walk(dir_path):
            for dir_name in dirs:
                dir_full_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(dir_full_path)
                    print(f"Deleted directory: {dir_full_path}")
                except Exception as e:
                    print(f"Error deleting directory {dir_full_path}: {e}")

# List of directories to delete
directories_to_delete = [
    '/mnt/recordings/clusters/1/',
    '/mnt/recordings/clusters/2/',
    '/mnt/recordings/clusters/3/'
]

# Walk through the directory and delete files while preserving directories
for root, dirs, files in os.walk(mount_point):
    for file in files:
        file_path = os.path.join(root, file)
        try:
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

delete_directories(directories_to_delete)
print("Cleanup completed.")