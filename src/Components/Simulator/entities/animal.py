

#############################################################################
# This class represents a simulated animal
#############################################################################

import species
import entity

class Animal(entity.Entity):
    def __init__(self) -> None:
        self.species = species.Species.DOG
        
