

#############################################################################
# This class represents a simulated animal
#############################################################################

import entities.species
import entities.entity
import datetime

class Animal(entities.entity.Entity):
    def __init__(self, lla=(0,0,10.0)) -> None:
        self.species = entities.species.Species.DOG
        self.sound_produced_time: datetime

        super(Animal, self).__init__(lla)
        self.set_random_lla()
        self.set_sound_production_time()

    def set_random_lla(self) -> None:
        x, y = self.randLatLong()
        self.lla = (x, y, 10.0)
    
    def set_sound_production_time(self) -> None:
        self.sound_produced_time = datetime.datetime.now()

    def get_sound_production_time(self) -> datetime:
        return self.sound_produced_time