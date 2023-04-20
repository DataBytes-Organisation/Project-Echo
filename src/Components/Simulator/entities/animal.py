
#############################################################################
# This class represents a simulated animal
#############################################################################

from clock import Clock
import entities.species
import entities.entity
import datetime
import numpy as np

class Animal(entities.entity.Entity):
    
    def __init__(self, species, 
                 lla=None, 
                 vocal_interval_mean=0.1,
                 vocal_interval_std=1.0) -> None:
        
        # call parent constructor
        super(Animal, self).__init__(lla)
        
        # get the singleton clock handle for time calls
        self.clock = Clock()
        
        # the animal species
        self.species = species
        
        # the mean interval for vocalisation
        self.vocal_interval_mean = vocal_interval_mean
        
        # the standard deviation for vocalisation
        self.vocal_interval_std = vocal_interval_std
        
        # initialise last vocal time to sim time
        self.last_vocalisation_time = self.clock.get_time()
        
        # interval for vocalisaiton will be sampled form a normal distribution
        # calculate random wait for next vocalisation
        self.next_vocal_random_wait = np.random.normal(
            self.vocal_interval_mean,
            self.vocal_interval_std,
            None)

    def describe(self) -> None:
        print(f'Animal Species        : {self.species}')
        print(f'Animal LLA            : {self.getLLA()}')
        print(f'Animal Vocal Mean (s) : {self.vocal_interval_mean}')
        print(f'Animal Vocal Std (s)  : {self.vocal_interval_std}')
        
    def update_lla(self) -> None:
        pass
    
    def random_vocalisation(self) -> None:
        print(f'Random sample for vocalisation')
        
        # get the current sim time
        sim_time = self.clock.get_time()
        
        # calculate how much sim time has elapsed since last vocal
        elapsed = (sim_time - self.last_vocalisation_time).total_seconds() 
        
        # determine if we have hit the threshold for this animal
        if elapsed > self.next_vocal_random_wait:
            
            # TODO: send the vocalisation
            print(f'simulated vocalisation from Animal {self} time: {sim_time}')
            
            # calculate when next vocalisation will occur
            self.next_vocal_random_wait = np.random.normal(
                self.vocal_interval_mean,
                self.vocal_interval_std,
                None) 

    def set_random_lla(self) -> None:
        x, y, a = self.randLatLong()
        self.lla = (x, y, a)
    
    def set_sound_production_time(self) -> None:
        self.sound_produced_time = datetime.datetime.now()

    def get_sound_production_time(self) -> datetime:
        return self.sound_produced_time