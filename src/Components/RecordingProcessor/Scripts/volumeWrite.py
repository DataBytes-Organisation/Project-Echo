import os

# Define the path to the Docker volume (replace with your actual path)
docker_volume_path = "/var/lib/docker/volumes/echo-net_recordings/_data/pre-processed"

# Ensure the directory exists
if not os.path.exists(docker_volume_path):
    raise Exception(f"Directory {docker_volume_path} does not exist. Make sure the Docker volume is mounted correctly.")

# Write a sample file to the Docker volume
file_path = os.path.join(docker_volume_path, "test_file.txt")
with open(file_path, "w") as file:
    file.write("Hello, this is a test file written to the Docker volume.\n")

print(f"File has been written to {file_path}")