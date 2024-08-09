import os

# Define the mount point
mount_point = "/var/lib/docker/volumes/echo-net_db-data"

# Verify the mount point
if not os.path.isdir(mount_point):
    raise Exception(f"Mount point {mount_point} is not a directory or does not exist")

print(f"Inspecting directories within the volume at {mount_point}")

# Walk through the directory and list all directories and files
for root, dirs, files in os.walk(mount_point):
    for dir in dirs:
        print(f"Directory: {os.path.join(root, dir)}")
    for file in files:
        print(f"File: {os.path.join(root, file)}")