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
    def __init__(self, lla=None) -> None:
        super().__init__(lla)

    def render_initial_sensor_state(self, config, animals):
        m = folium.Map(location=self.get_otways_coordinates()[0], zoom_start=13)
        for mic in config.SENSOR_MANAGER.MicrophoneObjects:
            folium.Marker(location=[mic.getLLA()[0], mic.getLLA()[1]], icon=folium.Icon(icon="microphone", prefix='fa', color="black"), custom_id=mic.unique_identifier).add_to(m)
        for animal in animals:
            folium.Marker(location=[animal.getLLA()[0], animal.getLLA()[1]], icon=folium.Icon(icon="dove", prefix="fa", color="green"), popup=str(str(animal.species) + '\n' + animal.uuid), custom_id=animal.uuid).add_to(m)

        m.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'RenderedState.html'))
        self.folium_map = m
    
    def render(self, animals):
        for animal in animals:
            for marker_ in self.folium_map._children.values():
                try:
                    if str(animal.uuid) == str(marker_.options['customId']):
                        marker_.location[0] = animal.getLLA()[0]
                        marker_.location[1] = animal.getLLA()[1]
                except: pass

        self.folium_map.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'RenderedState.html'))
    
    def render_animal_vocalisation(self, animal):
        for marker_ in self.folium_map._children.values():
            try:
                if str(animal.uuid) == str(marker_.options['customId']):
                    marker_.location[0] = animal.getLLA()[0]
                    marker_.location[1] = animal.getLLA()[1]
                    marker_.icon.options['markerColor'] = 'red'
            except: pass

        self.folium_map.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'RenderedState.html'))
        input('check audio visualisation')
                
class Simulator():
    def __init__(self) -> None:
        self.config = simulator_init.Config()
        self.clock  = Clock(step_interval=0.200) # 200 step interval 200 milliseconds

    # run the live simulators
    def execute(self):
        
        # initialse the simulator configuration
        animals = self.config.initialise()

        # render state
        if bool(self.config._get_config('RENDER_STATES')):
            self.render_state = RenderedState()
            self.render_state.render_initial_sensor_state(self.config, animals)
    
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
                    self.render_state.render_animal_vocalisation(animal)
                    # self.config.SENSOR_MANAGER.blah()
                    pass
                    # TODO need to process the sensors here

                animal.describe()
                
            # render state to map
            self.render_state.render(animals)
            
            # process API commands
            self.process_api_commands()
            
            # wait for wall clock to elapse to sync with real time
            self.wait_real_time_sync()

        
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
    