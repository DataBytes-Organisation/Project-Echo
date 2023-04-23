import os
import json
import unittest
import datetime
import logging
from datetime import datetime
from entities.microphone import MicrophoneStation
from sensor_manager import SensorManager
from factories.animal_factory import AnimalFactory
from factories.sensor_factory import SensorFactory
from comms_manager import CommsManager
from system_manager import SystemManager


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.config.initialise()

    def test_valid_logger(self):
        _sys_logger = logging.getLogger('_sys_logger')
        _sys_logger.debug('Running UnitTests')

        self.assertLogs(_sys_logger, level='DEBUG')

    def test_nodes_connected(self):
        graph = self.config.SENSOR_MANAGER.GRAPH
        for node in graph.nodes():
            assert len(graph.edges(node)) > 0, f"Node {node} has no edges"

    def test_valid_system_params(self):
        true_count = 0
        for var in ['SYSTEM_START', 'SYSTEM_PAUSE', 'SYSTEM_RESTART', 'SYSTEM_STOP']:
            if bool(os.environ[var]):
                true_count += 1
        
        assert true_count == 1, f'System Variables are INVALID. Only one can be true at once.'

    @unittest.skip
    def test_database_connection(self):
        print('no database connection string provided yet')
        pass


class Config:
    def __init__(self):
        self.config = self.read_configuration()

    def _get_config(self):
        return self.config
    
    def initialise(self):    
        # Validate the configuration
        self.validate_configuration(self._get_config)
        
        # Initialise logging file
        self.initiase_logging(self._get_config())
        
        # Create the communications manager
        self.comms_manager = CommsManager()

        # Create the system manager
        self.system_manager = SystemManager()
        
        # Initialise the communications manager
        self.comms_manager.initialise_communications()
        
        # Use google cloud to initialise species list
        species_list = self.comms_manager.gcp_load_species_list()
        
        # Create the factories
        animal_factory = self.create_animal_factory(species_list)
        sensor_factory = self.create_sensor_factory()
        
        # Create Sensor Instances
        sensor_instances = self.create_sensor_instances(sensor_factory)
        
        # Create Sensor Manager and populate with sensors
        self.create_sensor_manager(sensor_instances)
        
        # Create Sensor Graph
        self.create_sensor_graph()
        
        # Create Animal Instances
        animal_instances = self.create_animal_instances(animal_factory)
        
        return animal_instances
    
    def read_configuration(self):
        config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', '.config')
        with open(config_file_path) as f:
            config_data = f.readlines()

        for line in [line for line in config_data if line != "\n"]:
            key, value = line.strip().split('=')
            os.environ[key] = value   
        return config_data        
    
    def validate_configuration(self, config):
        # TODO
        pass
    
    def initiase_logging(self, config):
         
        debug_level = 'INFO'
        if 'DEBUG_LEVEL' in os.environ:
            debug_level = os.environ['DEBUG_LEVEL']
            if debug_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                print(f"Invalid value '{debug_level}' for DEBUG_LEVEL in config file")
        else:
            print("DEBUG_LEVEL not set in config file")    
            
        _path_to_logger = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', '')
        if not os.path.exists(_path_to_logger):
            os.mkdir(_path_to_logger)

        _main_log_filename = _path_to_logger+'sys_log_'+datetime.now().date().isoformat()+'_.log'

        logger1 = logging.getLogger('_sys_logger')
        handler1 = logging.FileHandler(filename=_main_log_filename, mode="w", encoding='utf-8')
        handler1.setLevel(debug_level)
        handler1.setFormatter(logging.Formatter('%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S'))
        logging.getLogger('_sys_logger').addHandler(handler1)
        logger1.setLevel(debug_level)               
    
    def create_animal_factory(self, species_list):
        factory = AnimalFactory(species_list)
        return factory
    
    def create_sensor_factory(self):
        factory = SensorFactory()
        return factory
    
    def create_sensor_instances(self, sensor_factory):
        instances = []
        _path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'mics_info.json')
        with open(_path, 'r') as f:
            mics_info = json.load(f)
        for mic_info in mics_info:
            instance = sensor_factory.create(mic_info["unique_identifier"], 
                                  mic_info["name"], 
                                  (mic_info["latitude"], mic_info["longitude"], mic_info["elevation"]))

            instances.append(instance)
        return instances
        
    def create_sensor_manager(self, sensor_instances):
        self.SENSOR_MANAGER = SensorManager()
        for instance in sensor_instances:
            self.SENSOR_MANAGER.add_sensor_object(instance)

    def create_sensor_graph(self):
        self.SENSOR_MANAGER.generate_sensor_graph()

    def create_animal_instances(self, animal_factory):
        instances = []
        # TODO: we should use config for this
        for a in range(2):
            animal = animal_factory.create()
            lla = animal.randLatLong()
            animal.setLLA(lla)
            animal.describe()
            instances.append(animal)
        return instances
        

if __name__ == "__main__":
    print('Model to be imported, not run')
