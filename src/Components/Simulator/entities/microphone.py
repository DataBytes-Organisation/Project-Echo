
#############################################################################
# This class represents a simulated microphone station
#############################################################################

import entities.entity
import uuid

class MicrophoneStation(entities.entity.Entity):
    def __init__(self, lla=(0,0,10.0)) -> None:
        self.name = "OnField-Microphone"
        self.unique_identifier = str(uuid.uuid4())
        self.time_delay_seconds: float
        super(MicrophoneStation, self).__init__(lla)

    def set_time_delay(self, tds) -> None:
        self.time_delay_seconds = tds

    def get_time_delay(self) -> None:
        return self.time_delay_seconds
        

        