
#############################################################################
# This class represents a simulated microphone station
#############################################################################

import entities.entity

class MicrophoneStation(entities.entity.Entity):
    def __init__(self, lla=(-38.0,134.0,10.0)) -> None:
        self.name = "Station 1"
        super(MicrophoneStation, self).__init__(lla)
        

        