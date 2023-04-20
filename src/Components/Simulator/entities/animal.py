

#############################################################################
# This class represents a simulated animal
#############################################################################

import entities.species
import entities.entity
import datetime

class Animal(entities.entity.Entity):
    
    def __init__(self, species, 
                 lla=None, 
                 vocal_interval_mean=0.1,
                 vocal_interval_std=1.0) -> None:
        
        # call parent constructor
        super(Animal, self).__init__(lla)
        
        # the animal species
        self.species = species
        
        # the mean interval for vocalisation
        self.vocal_interval_mean = vocal_interval_mean
        
        # the standard deviation for vocalisation
        # This controls the 'randomness'
        self.vocal_interval_std = vocal_interval_std
        
        self.last_vocalisation_time = : datetime

        
        self.set_random_lla()
        self.set_sound_production_time()
        
    def describe(self) -> None:
        print(f'Animal Species : {self.species}')
        print(f'Animal LLA     : {self.getLLA()}')
        print(f'Animal Vocal Mean (s) : {self.vocal_interval_mean}')
        print(f'Animal Vocal Std (s) : {self.vocal_interval_std}')
        
    def update_lla(self) -> None:
        pass
    
    def random_vocalisation(self) -> None:
        pass    

    def set_random_lla(self) -> None:
        x, y, _ = self.randLatLong()
        self.lla = (x, y, 10.0)
    
    def set_sound_production_time(self) -> None:
        self.sound_produced_time = datetime.datetime.now()

    def get_sound_production_time(self) -> datetime:
        return self.sound_produced_time