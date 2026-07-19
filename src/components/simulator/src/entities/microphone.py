
#############################################################################
# This class represents a simulated microphone station
#############################################################################

import entities.entity
from clock import Clock

class MicrophoneStation(entities.entity.Entity):
    def __init__(self, _uuid, name, lla) -> None:
        self.name = name
        self.unique_identifier = _uuid

        super(MicrophoneStation, self).__init__(lla)

        # get the singleton clock handle for time calls
        self.clock = Clock()
        self.clock_time = self.clock.get_time()

    def set_trigger_event_time(self, clock_time) -> None:
        self.triggered_sim_clock_time = clock_time

    def reset_trigger_event_time(self) -> None:
        self.triggered_sim_clock_time = self.clock.get_time()
        
    def getID(self):
        return self.unique_identifier