import simulator_init
import unittest
from clock import Clock

from factories.animal_factory import AnimalFactory
from factories.sensor_factory import SensorFactory

class TestConfig(simulator_init.TestConfig):
    def __init__(self, *args, config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
                
class Simulator():
    def __init__(self) -> None:
        self.config = simulator_init.Config()
        self.clock  = Clock()
        
    # run the live simulator
    def execute(self):
        
        # initialse the simulator configuration
        (animals, sensors) = self.config.initialise()
        
        # start the simulator loop
        self.main_loop(animals, sensors)
        
    def main_loop(self, animals, sensors):
        # update the simulated time
        
        for animal in animals:
            # update the animal lla
            animal.update_lla()
            
            # generate random animal vocalisation
            animal.random_vocalisation()
            
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
        # TODO
        pass       

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
        print(self.AnimalFactory.create_random_animal().species)

if __name__ == "__main__":
    sim = Simulator()
    sim.test()
    