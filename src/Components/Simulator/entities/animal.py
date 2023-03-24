

#############################################################################
# This class represents a simulated animal
#############################################################################

import entities.species
import entities.entity

class Animal(entities.entity.Entity):
    def __init__(self, lla=(0,0,10.0)) -> None:
        self.species = entities.species.Species.DOG
        super(Animal, self).__init__(lla)
        self.set_random_lla()

    def set_random_lla(self) -> None:
        x, y = self.randLatLong()
        self.lla = (x, y, 10.0)
        
