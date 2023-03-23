

#############################################################################
# This class represents a simulated animal
#############################################################################

import entities.species
import entities.entity

class Animal(entities.entity.Entity):
    def __init__(self, lla=(-38.0,134.0,10.0)) -> None:
        self.species = entities.species.Species.DOG
        super(Animal, self).__init__(lla)
        
