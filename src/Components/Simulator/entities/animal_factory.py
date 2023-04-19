import random
from entities.animal import Animal
from entities.species import Species
import entities.entity

class AnimalFactory():
    @staticmethod
    def create_random_animal(lla=entities.entity.Entity((0,0,10)).randLatLong()):
        species = random.choice(list(Species))
        return Animal(species, lla)