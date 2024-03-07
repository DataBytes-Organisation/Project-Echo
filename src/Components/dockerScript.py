import docker
import subprocess
import argparse



# List of container names to delete
containers_to_delete = ["ts-echo-model-cont","ts-echo-hmi-cont","ts-api-cont", "ts-simulator-cont", "ts-echo-engine-cont", "ts-mongodb-cont", "mongo-express", "ts-mqtt-server-cont","echo-redis"]
images = ["ts-simulator","ts-api","ts-mongodb","ts-echo-hmi","ts-echo-engine","ts-mqtt-server","ts-echo-model","mongo-express", "redis"]
preserved_volumes = ["echo-net_credentials_volume", "echo-net_db-data"]

containers_to_deleteV2 = []
imagesV2 = []
preserved_volumesV2 = []

def deletionList(containers):
    if containers == "hmi":
        containers_to_deleteV2.append("ts-echo-hmi-cont")
        imagesV2.append("ts-echo-hmi")

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
    if containers == "hmi":
        try:
            subprocess.run(["docker-compose", "-f", "docker-compose.hmi.yml", "up", "--build"])
        except subprocess.CalledProcessError as e:
            print(f"Error running the docker-compose.hmi.yaml file")
# Create the parser
parser = argparse.ArgumentParser(description='Process some integers.')

# Add the arguments
parser.add_argument('-cont', '--container', type=str, help='The name of the container')

# Execute the parse_args() method
containers = parser.parse_args()

print(f"Container Name: {containers.container}")

if __name__ == "__main__":

    if containers.container is not None:
        print(f"Container name provided: {containers.container}. Execute partial rebuild...")
        deletionList(containers.container)
        delete_containers(containers_to_deleteV2)
        delete_images(imagesV2)
        compose(containers.container)
        

    else:
        print("No container name provided, execute complete rebuild...")
        delete_containers(containers_to_delete)
        delete_images(images)
        delete_unused_volumes(preserved_volumes)
        run_docker_compose_up()
