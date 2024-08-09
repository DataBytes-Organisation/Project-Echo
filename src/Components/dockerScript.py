import docker
import subprocess
import argparse

# List of container names to delete
containers_to_delete = ["ts-echo-model-cont","ts-echo-hmi-cont","ts-api-cont", "ts-simulator-cont", "ts-echo-engine-cont", "ts-mongodb-cont", "mongo-express", "ts-mqtt-server-cont","echo-redis","ts-triangulation-cont"]
images = ["ts-simulator","ts-api","ts-mongodb","ts-echo-hmi","ts-echo-engine","ts-mqtt-server","ts-echo-model","mongo-express", "redis","ts-echo-triangulation"]
preserved_volumes = ["echo-net_credentials_volume", "echo-net_db-data", "echo-net_recordings"]

containers_to_deleteV2 = []
imagesV2 = []
preserved_volumesV2 = []

def deletionList(containers):
    for container in containers:

        if container == "hmi":
            print("Removing hmi...")
            containers_to_deleteV2.append("ts-echo-hmi-cont")
            imagesV2.append("ts-echo-hmi")

        elif container == 'eng':
            print("Removing engine...")
            containers_to_deleteV2.append("ts-echo-engine-cont")
            imagesV2.append("ts-echo-engine")

        elif container == 'sim':
            print("Removing simulator...")
            containers_to_deleteV2.append("ts-simulator-cont")
            imagesV2.append("ts-simulator")

        elif container == 'api':
            print("Removing api...")
            containers_to_deleteV2.append("ts-api-cont")
            imagesV2.append("ts-api")
        
        elif container == "tri":
            containers_to_deleteV2.append("ts-triangulation-cont")
            imagesV2.append("ts-echo-triangulation")
        
        elif container == "onset":
            containers_to_deleteV2.append('ts-onset-cont')
            imagesV2.append("ts-echo-onset")

        elif container == "clustering":
            containers_to_deleteV2.append('ts-clustering-cont')
            imagesV2.append("ts-echo-clustering")
        



def delete_containers(container_names):
    client = docker.from_env()

    for container_name in container_names:
        try:
            container = client.containers.get(container_name)
            container.stop()
            container.remove()
            print(f"Container {container_name} stopped and deleted.")
        except docker.errors.NotFound:
            print(f"Container {container_name} not found.")


def delete_unused_volumes(preserved_volume_names):
    client = docker.from_env()

    # Get a list of all volumes
    all_volumes = client.volumes.list()

    for volume in all_volumes:
        if volume.name not in preserved_volume_names:
            try:
                volume.remove()
                print(f"Volume {volume.name} deleted.")
            except docker.errors.APIError as e:
                print(f"Error deleting volume {volume.name}: {e}")

def delete_images(image_names_or_ids):
    client = docker.from_env()

    for name_or_id in image_names_or_ids:
        try:
            image = client.images.get(name_or_id)
            client.images.remove(image.id)
            print(f"Image {name_or_id} deleted.")
        except docker.errors.ImageNotFound:
            print(f"Image {name_or_id} not found.")

def run_docker_compose_up():
    try:
        subprocess.run(["docker-compose", "up", "--build"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running 'docker-compose up --build': {e}")

def compose(containers):
    command = ["docker-compose"]
    print("Composing containers:    ", containers)
    
    # Dynamically add all compose files to the command
    for container in containers:
        compose_file = f"docker-compose.{container}.yml"
        command.extend(["-f", compose_file])
    
    # Add the 'up' and '--build' commands at the end
    command.extend(["up", "--build"])
    
    # Print the constructed command for debugging
    print("Constructed command: ", " ".join(command))
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running docker-compose with files for {containers}: {e}")


# Create the parser
parser = argparse.ArgumentParser(description='Process some integers.')

# Add the arguments
parser.add_argument('-cont', '--container', nargs='+', type=str, help='The names of the containers')


# Execute the parse_args() method
containers = parser.parse_args()

for c in containers.container:
    print(f"Container Name: {c}")



if __name__ == "__main__":
    


    if containers.container is not None:
        print(f"Container name provided: {containers.container}. Execute partial rebuild...")
        deletionList(containers.container)
        delete_containers(containers_to_deleteV2)
        delete_images(imagesV2)
        print(containers.container)
        compose(containers.container)
        

    else:
        print("No container name provided, execute complete rebuild...")
        delete_containers(containers_to_delete)
        delete_images(images)
        delete_unused_volumes(preserved_volumes)
        run_docker_compose_up()
