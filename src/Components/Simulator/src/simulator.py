import simulator_init
import os
import unittest
from clock import Clock
import ast
import logging
logger1 = logging.getLogger('_sys_logger')

from factories.animal_factory import AnimalFactory
from factories.sensor_factory import SensorFactory
from render_manager import RenderedState
import asyncio

class TestConfig(simulator_init.TestConfig):
    def __init__(self, *args, config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

class Simulator():
    def __init__(self) -> None:
        self.config = simulator_init.Config()
        self.clock  = Clock(step_interval=float(os.environ['STEP_INTERVAL'])) # 200 step interval 200 milliseconds

    # run the live simulators
    async def execute(self):
        # initialse the simulator configuration
        animals, sensors = self.config.initialise()
        
        # send the list of microphones across to the database
        self.config.comms_manager.echo_api_set_microphones(sensors)

        # render state
        if ast.literal_eval(os.environ['RENDER_STATES']):
            self.render_state = RenderedState()
            self.render_state.render_initial_sensor_state(self.config, animals)
    
        # start the simulator loop
        await self.main_loop(animals, loops=int(os.environ['SIMULATOR_LOOPS']))
        
    async def main_loop(self, animals, loops=100000):
        for _ in range(loops):
            print("Simulator Loop....", flush=True)
            try:
                # we need this - dont delete
                await asyncio.sleep(0)
                print(self.clock.get_time())

                # update the simulated time (advance the clock)
                self.clock.update()
                
                for animal in animals:
                    print("Animals Loop....", flush=True)
                    # update the animal lla
                    animal.update_lla()
                    
                    # send animal movement event for debugging
                    self.config.comms_manager.echo_api_send_animal_movement(animal)
                    
                    # generate random animal vocalisation
                    if animal.random_vocalisation():
                        print("Animal Vocal....", flush=True)
                        if ast.literal_eval(os.environ['RENDER_STATES']):  self.render_state.render_animal_vocalisation(animal)
                        predicted_lla, closest_mic, min_error = self.config.SENSOR_MANAGER.vocalisation(animal)
           
                        self.config.comms_manager.mqtt_send_random_audio_msg(animal, predicted_lla, closest_mic, min_error)
                        print("Animal Vocal....Complete.", flush=True)

                    animal.describe()
                
                # render state to map
                if ast.literal_eval(os.environ['RENDER_STATES']): self.render_state.render(animals)
                
                # process API commands
                self.process_api_commands()
                
                # wait for wall clock to elapse to sync with real time
                self.wait_real_time_sync()
            except asyncio.CancelledError:
                logger1.critical('Asyncio error formed')
                break
            except Exception as e:
                print(f"An error occurred: {e}", flush=True)
            finally:
                pass
                

    def process_api_commands(self):
        # TODO
        pass
    
    def wait_real_time_sync(self):
        self.clock.wait_real_time_sync()
      
    # run some simulator test cases
    def test(self):
        suite = unittest.TestSuite()
        for test_name in unittest.defaultTestLoader.getTestCaseNames(TestConfig):
            suite.addTest(TestConfig(test_name, config=self.config))

        unittest_runner = unittest.TextTestRunner()
        unittest_runner.run(suite)

        self.SystemClock = Clock()
        self.AnimalFactory = AnimalFactory()
        self.SensorFactory = SensorFactory()
        logger1.info(f'Random animal create(): {self.AnimalFactory.create().species}')
        logger1.info(f'Random animal create_random_animal(): {self.AnimalFactory.create_random_animal().species}')

if __name__ == "__main__":
    print('Meant to be imported by System Manager - not run otherwise cant be controlled.')
    # sim = Simulator()
    # sim.execute()
    
    