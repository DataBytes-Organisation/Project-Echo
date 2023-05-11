import os
import gc
import asyncio
from asyncio_mqtt import Client as MqttClient
from asyncio_mqtt import Topic
import simulator

class SystemManager:
    def __init__(self):
        self.read_configuration()
        self.sim_running = False
        self.sim_task = None
        self.command_queue = asyncio.Queue()

    def read_configuration(self):
        config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', '.config')
        with open(config_file_path) as f:
            config_data = f.readlines()

        for line in [line for line in config_data if line != "\n"]:
            key, value = line.strip().split('=')
            os.environ[key] = value

    async def initialise_communications(self):
        # sleep and wait for MQTT server to initialise
        print("Initialising communications with MQTT", flush=True)
        connected = False
        while not connected:
            try:
                async with MqttClient(os.environ['MQTT_CLIENT_URL'], int(os.environ['MQTT_CLIENT_PORT']), clean_session=True) as mqtt_client:
                    print("Connected... waiting for start command", flush=True)
                    connected = True
                    await mqtt_client.subscribe("Simulator_Controls")
                    await asyncio.gather(self.handle_messages(mqtt_client), self.run_loop())
                              
            except Exception as e:
                print(f"Exception {e} Retrying...", flush=True)
            await asyncio.sleep(1)    

    async def handle_messages(self, mqtt_client):
        topic_filter = "Simulator_Controls"
        async with mqtt_client.messages() as messages:
            async for msg in messages:
                if str(topic_filter) == str(msg.topic):
                    await self.on_message(mqtt_client, None, msg)

    async def on_message(self, client, userdata, msg):
        message = msg.payload.decode('utf-8')
        await self.command_queue.put(message)

    async def run_loop(self):
        while True:
            command = await self.command_queue.get()
            if command == "Start":
                if self.sim_running:
                    print("Simulator is already running", flush=True)
                else:
                    print("Starting Simulator....", flush=True)
                    self.sim_task = asyncio.create_task(self.run_sim())
                    self.sim_running = True
                    print("Simulator Running....", flush=True)
            elif command == "Stop":
                if not self.sim_running:
                    print("Simulator is not running", flush=True)
                else:
                    print("Stopping Simulator", flush=True)
                    self.sim_task.cancel()
                    self.sim_running = False
                    print("Simulator has stopped.", flush=True)

    async def run_sim(self):
        self.simulator = simulator.Simulator()
        await self.simulator.execute()

async def main():
    system_manager = SystemManager()
    await system_manager.initialise_communications()

if __name__ == '__main__':
    asyncio.run(main())
