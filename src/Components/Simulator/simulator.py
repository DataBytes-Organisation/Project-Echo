import simulator_init
import os
import unittest
from clock import Clock
import folium
from entities.entity import Entity

from factories.animal_factory import AnimalFactory
from factories.sensor_factory import SensorFactory

class TestConfig(simulator_init.TestConfig):
    def __init__(self, *args, config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

class RenderedState(Entity):
    def __init__(self, config, lla=None) -> None:
        super().__init__(lla)
        self.config = config
        self.render_initial_sensor_state()

    def render_initial_sensor_state(self):
        m = folium.Map(location=self.get_otways_coordinates()[0], zoom_start=13)
        for mic in self.config.SENSOR_MANAGER.MicrophoneObjects:
            folium.Marker(location=[mic.getLLA()[0], mic.getLLA()[1]], icon=folium.Icon(icon="microphone", prefix='fa', color="orange")).add_to(m)
        m.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'RenderedState.html'))
        self.folium_map = m
                
class Simulator():
    def __init__(self) -> None:
        self.config = simulator_init.Config()
        print(self.config.SENSOR_MANAGER.MicrophoneObjects)
        if bool(self.config._get_config('RENDER_STATES')):
            self.render_state = RenderedState(self.config)
            input('check map output')
        self.clock  = Clock(step_interval=0.200) # 200 step interval 200 milliseconds

    # run the live simulator
    def execute(self):
        
        # initialse the simulator configuration
        animals = self.config.initialise()
    
        # start the simulator loop
        self.main_loop(animals, loops=10)
        
    def main_loop(self, animals, loops=10):
        
        for loop in range(loops):
        
            # update the simulated time (advance the clock)
            self.clock.update()
            
            for animal in animals:
                
                # update the animal lla
                animal.update_lla()
                
                # generate random animal vocalisation
                if animal.random_vocalisation():
                    # self.config.SENSOR_MANAGER.blah()
                    pass
                    # TODO need to process the sensors here

                animal.describe()
                
            # render state to map
            self.render_state_to_map()
            
            # process API commands
            self.process_api_commands()
            
            # wait for wall clock to elapse to sync with real time
            self.wait_real_time_sync()
           
    def render_state_to_map(self):
        # TODO
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
        print(f'Random animal create(): {self.AnimalFactory.create().species}')
        print(f'Random animal create_random_animal(): {self.AnimalFactory.create_random_animal().species}')

if __name__ == "__main__":
    
    #clock = Clock()
    #clock.test()
    
    sim = Simulator()
    #sim.test()
    
    # by default it will run 10 loops
    sim.execute()
    