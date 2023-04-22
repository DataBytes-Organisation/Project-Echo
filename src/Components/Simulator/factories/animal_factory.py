import random
from entities.animal import Animal
from entities.species import Species


# implementation of the Singleton Pattern for Factories
class AnimalFactory(object):
    _instance = None

    def __new__(cls, species_list):
        if cls._instance is None:
            print('Creating the AnimalFactory Once')
            cls._instance = super(AnimalFactory, cls).__new__(cls)
            cls.species_list = species_list
            # Put any initialization here.
        return cls._instance
    
    def create_random_animal(self):
        species = random.choice(self.species_list)
        return Animal(species)    
    
    def create(self):
        # create a random one for now
        return self.create_random_animal()
