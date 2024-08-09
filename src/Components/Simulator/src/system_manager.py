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
        self.sim_mode = "Animal Mode"

        self.topic_handlers = {
            str("Simulator_Controls"): self.on_message,
            str("Simulate_Recording"): self.on_recording_message
        }

    def read_configuration(self):
        config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', '.config')
        with open(config_file_path) as f:
            config_data = f.readlines()

        for line in [line for line in config_data if line != "\n"]:
            key, value = line.strip().split('=')
            os.environ[key] = value



    async def initialise_communications(self):
        print("Initialising communications with MQTT", flush=True)
        while True:
            tasks = []
            try:
                async with MqttClient(os.environ['MQTT_CLIENT_URL'], int(os.environ['MQTT_CLIENT_PORT']), clean_session=True) as mqtt_client:
                    print("Connected... waiting for start command", flush=True)
                    await mqtt_client.subscribe("Simulator_Controls")
                    await mqtt_client.subscribe("Simulate_Recording")
                    tasks = [
                        asyncio.create_task(self.handle_messages(mqtt_client)),
                        asyncio.create_task(self.run_loop())
                    ]
                    await asyncio.gather(*tasks)
            except Exception as e:
                print(f"Exception in communications: {e}", flush=True)
                for task in tasks:
                    task.cancel()
                try:
                    await asyncio.gather(*tasks)
                except asyncio.CancelledError:
                    print("All tasks cancelled after an error.", flush=True)
            finally:
                await asyncio.sleep(1)

    async def handle_messages(self, mqtt_client):
        async with mqtt_client.messages() as messages:
            async for msg in messages:
                topic = str(msg.topic)
                handler = self.topic_handlers.get(topic)
                if handler:
                    await handler(mqtt_client, None, msg)
                else:
                    print(f"Unhandled topic: {topic} - Available handlers: {list(self.topic_handlers.keys())}")
            ###THE MERGED CODE IDK IF ITS RIGHT
                if str(topic_filter) == str(msg.topic):
                    await self.on_message(mqtt_client, None, msg)
                if str(topic_filter2) == str(msg.topic):
                    await self.on_recording_message(mqtt_client, None, msg)
                
    async def on_message(self, client, userdata, msg):
        message = msg.payload.decode('utf-8')
        await self.command_queue.put(message)
        print(f"Received command: {message}")

    async def on_recording_message(self, client, userdata, msg):
        print("Handling recording message")
        await self.simulator.handle_recording_message(msg)

    async def start_sim(self, mode):
        if self.sim_running:
            print("Simulator is already running", flush=True)
        else:
            print("Starting Simulator....", flush=True)
            self.sim_task = asyncio.create_task(self.run_sim(mode))
            self.sim_running = True
            print("Simulator Running....", flush=True)

    async def stop_sim(self):
        if not self.sim_running:
            print("Simulator is not running", flush=True)
        else:
            print("Stopping Simulator", flush=True)
            if self.sim_task:
                self.sim_task.cancel()
                try:
                    await self.sim_task
                except asyncio.CancelledError:
                    print("Simulator task cancelled.", flush=True)
                except Exception as e:
                    print(f"Error while stopping the simulator: {e}", flush=True)
            self.sim_running = False
            print("Simulator has stopped.", flush=True)

    async def restart_sim(self, mode):
        if self.sim_running:
            await self.stop_sim()
            await self.start_sim(mode)

    async def run_loop(self):
        while True:
            command = await self.command_queue.get()
            print(f"Simulator got command {command}", flush=True)
            
            if str(command) == str("Start"):
                await self.start_sim()

            elif str(command) == str("Stop"):
                await self.stop_sim()

            elif str(command) == str("Animal_Mode"):
                if self.sim_running:
                    await self.restart_sim(command)
                else:
                    await self.start_sim(command)

            elif str(command) == str("Recording_Mode"):
                if self.sim_running:
                    await self.restart_sim(command)
                else:
                    await self.start_sim(command)

            elif str(command) == str("Recording_Mode_V2"):
                if self.sim_running:
                    await self.restart_sim(command)
                else:
                    await self.start_sim(command)
                    
    async def run_sim(self, mode):
        self.simulator = simulator.Simulator()
        await self.simulator.execute(mode)
        #await self.simulator.set_mode(mode)

async def main():
    system_manager = SystemManager()
    await system_manager.initialise_communications()

if __name__ == '__main__':
    try:
        asyncio.run(main())
        print(f'Simulator Finished', flush=True)
    except Exception as e:
        print(f'Simulator Exception {e}', flush=True)
