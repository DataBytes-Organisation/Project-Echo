import docker
import subprocess

# List of container names to delete
containers_to_delete = ["ts-echo-model-cont","ts-echo-hmi-cont","ts-api-cont", "ts-simulator-cont", "ts-echo-engine-cont", "ts-mongodb-cont", "mongo-express", "ts-mqtt-server-cont"]
images = ["ts-simulator","ts-api","ts-mongodb","ts-echo-hmi","ts-echo-engine","ts-mqtt-server","ts-echo-model","mongo-express"]
preserved_volumes = ["echo-net_credentials_volume", "echo-net_db-data"]

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

if __name__ == "__main__":
    delete_containers(containers_to_delete)
    delete_images(images)
    delete_unused_volumes(preserved_volumes)
    run_docker_compose_up()