import random
from entities.animal import Animal
from entities.species import Species
import entities.entity


# implementation of the Singleton Pattern for Factories
class AnimalFactory(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print('Creating the AnimalFactory')
            cls._instance = super(AnimalFactory, cls).__new__(cls)
            # Put any initialization here.
        return cls._instance
    
    def create_random_animal(self):
        species = random.choice(list(Species))
        return Animal(species)    
    
    def create(self):
        # create a random one for now
        return self.create_random_animal()
