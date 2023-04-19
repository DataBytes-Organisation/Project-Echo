import random
from entities.animal import Animal
from entities.species import Species
import entities.entity

class AnimalFactory():
    @staticmethod
    def create_random_animal():
        species = random.choice(list(Species))
        return Animal(species)