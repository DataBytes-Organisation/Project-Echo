import os
import gc
import asyncio
from asyncio_mqtt import Client as MqttClient
from asyncio_mqtt import Topic
import simulator

class SystemManager:
    def __init__(self):
        self.read_configuration()
        self.simulator = None
        self.simulator_running = False
        self.simulator_task = None
        self.command_queue = asyncio.Queue()

    def read_configuration(self):
        config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', '.config')
        with open(config_file_path) as f:
            config_data = f.readlines()

        for line in [line for line in config_data if line != "\n"]:
            key, value = line.strip().split('=')
            os.environ[key] = value

    async def initialise_communications(self):
        async with MqttClient(os.environ['MQTT_CLIENT_URL'], int(os.environ['MQTT_CLIENT_PORT'])) as mqtt_client:
            await mqtt_client.subscribe("Simulator_Controls")
            await asyncio.gather(self.handle_messages(mqtt_client), self.run_loop())

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
            print('waiting for next command', flush=True)
            command = await self.command_queue.get()
            print(command)
            if command == "Start":
                print('Simulator start', flush=True)
                self.sim_task = asyncio.create_task(self.run_sim())
                print('Simulator started', flush=True)
            if command == "Stop":
                print('Simulator stop', flush=True)
                self.sim_task.cancel()

    async def run_sim(self):
        self.simulator = simulator.Simulator()
        await self.simulator.execute()

async def main():
    system_manager = SystemManager()
    await system_manager.initialise_communications()

if __name__ == '__main__':
    asyncio.run(main())
