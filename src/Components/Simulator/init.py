import os
import json
import unittest
import datetime
import logging
from datetime import datetime
from entities.microphone import MicrophoneStation
from entities.sensor_manager import SensorManager


class TestConfig(unittest.TestCase):
    def setUp(self):
        # Set up the Config object and read the environment variables
        self.config = Config()

    def test_valid_logger(self):
        _sys_logger = logging.getLogger('_sys_logger')
        _sys_logger.debug('Running UnitTests')

        self.assertLogs(_sys_logger, level='DEBUG')

    def test_nodes_connected(self):
        graph = self.config.SENSOR_MANAGER.GRAPH
        for node in graph.nodes():
            assert len(graph.edges(node)) > 0, f"Node {node} has no edges"


class Config:
    def __init__(self):
        # Read the contents of the config file
        config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', '.config')
        with open(config_file_path) as f:
            config_data = f.readlines()

        # Loop through each line in the config file and set the system variable
        for line in [line for line in config_data if line != "\n"]:
            key, value = line.strip().split('=')
            os.environ[key] = value
        
        # Set up the logger based on the value of DEBUG_LEVEL in the config file
        if 'DEBUG_LEVEL' in os.environ:
            debug_level = os.environ['DEBUG_LEVEL']
            if debug_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                self._create_logger(debug_level)
            else:
                print(f"Invalid value '{debug_level}' for DEBUG_LEVEL in config file")
        else:
            print("DEBUG_LEVEL not set in config file")

        self.SENSOR_MANAGER = SensorManager()
        self._load_mics_in()

    def _create_logger(self, debug_level):
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
 
    def _load_mics_in(self):
        _path_to_logger = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'mics_info.json')

        with open(_path_to_logger, 'r') as f:
            mics_info = json.load(f)

        for mic_info in mics_info:
            self.SENSOR_MANAGER.add_sensor_object(MicrophoneStation(mic_info["unique_identifier"], mic_info["name"], (mic_info["latitude"], mic_info["longitude"], mic_info["elevation"])))

        self.SENSOR_MANAGER.generate_sensor_graph()

if __name__ == "__main__":
    # Run the test case
    unittest.main()
