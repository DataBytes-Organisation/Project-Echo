import random
from entities.microphone import MicrophoneStation
import entities.entity
import logging
logger1 = logging.getLogger('_sys_logger')


# implementation of the Singleton Pattern for Factories
class SensorFactory(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger1.info('Creating the SensorFactory Once')
            cls._instance = super(SensorFactory, cls).__new__(cls)
            # Put any initialization here.
        return cls._instance
      
    def create(self, _uuid, name, lla):
        instance = MicrophoneStation(_uuid, name, lla)
        return instance
    
