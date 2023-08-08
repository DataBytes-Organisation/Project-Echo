import docker
# List of container names to delete
containers_to_delete = ["ts-echo-model-cont","ts-echo-hmi-cont","ts-api-cont", "ts-simulator-cont", "ts-echo-engine-cont", "ts-mongodb-cont", "mongo-express", "ts-mqtt-server-cont"]
images = ["ts-simulator","ts-api","ts-mongodb","ts-echo-hmi","ts-echo-engine","ts-mqtt-server","ts-echo-model","mongo-express"]
volumes = ["2c1f963fada457ab47a518d8b2a45c390d1996eb664f887dafd24efe219e92a7", "cd2a47b2d930fced0ccff307993cb1fa97a5e380ca95c21a06acb9ea4e1b816a", "e8bd80fc599b70c1bbcea0a551f45eb5da995954ec02f3d5ee4a9a3a16fcd322", "echo-net_db-data"]
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

credentials = "echo-net_credentials_volume"

def delete_unused_volumes(preserved_volume_name):
    client = docker.from_env()

    # Get a list of all volumes
    all_volumes = client.volumes.list()

    for volume in all_volumes:
        if volume.name != preserved_volume_name:
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

if __name__ == "__main__":
    delete_containers(containers_to_delete)
    delete_images(images)
    delete_unused_volumes(credentials)

