
#############################################################################
# This class represents a simulated microphone station
#############################################################################

import entities.entity
import datetime

class MicrophoneStation(entities.entity.Entity):
    def __init__(self, _uuid, name, lla) -> None:
        self.name = name
        self.unique_identifier = _uuid

        self.event_timestamp: datetime
        self.TRIGGERED: bool = False
        super(MicrophoneStation, self).__init__(lla)

    def set_trigger_event_time(self, TT: datetime) -> None:
        self.event_timestamp = TT
        self.set_trigger()
    
    def get_trigger_event_time(self) -> datetime:
        return self.event_timestamp
    
    def set_trigger(self) -> None:
        self.TRIGGERED = True
    
    def reset(self) -> None:
        self.TRIGGERED = False
        self.event_timestamp = None
        

        