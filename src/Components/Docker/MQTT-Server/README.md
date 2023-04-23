Run docker-compose up -d in your terminal to build the Docker image and start the HiveMQ MQTT broker.
Once the HiveMQ MQTT broker is running, you can use the following URLs to interact with it:

To send and receive messages, connect your MQTT client to mqtt://<your-machine-ip>:1883. Replace <your-machine-ip> with your machine's IP address or localhost if running the broker on your local machine.
To access the HiveMQ Control Center, open a web browser and navigate to http://<your-machine-ip>:8000. Replace <your-machine-ip> with your machine's IP address or localhost if running the broker on your local machine.