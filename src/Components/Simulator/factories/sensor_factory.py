import random
from entities.microphone import MicrophoneStation
import entities.entity


# implementation of the Singleton Pattern for Factories
class SensorFactory(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print('Creating the SensorFactory')
            cls._instance = super(SensorFactory, cls).__new__(cls)
            # Put any initialization here.
        return cls._instance
      
    def create(self, _uuid, name, lla):
        instance = MicrophoneStation(_uuid, name, lla)
        return instance