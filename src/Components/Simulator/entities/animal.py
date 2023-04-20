
#############################################################################
# This class represents a simulated animal
#############################################################################

from clock import Clock
import entities.species
import entities.entity
import numpy as np
import base64
import uuid

class Animal(entities.entity.Entity):
    
    def __init__(self, species, 
                 lla=None, 
                 vocal_interval_mean=0.1,
                 vocal_interval_std=1.0) -> None:
        
        # call parent constructor
        super(Animal, self).__init__(lla)
        
        # get the singleton clock handle for time calls
        self.clock = Clock()
        
        # allocate a uuid
        self.uuid = self.short_uuid()
        
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
        print(f'Animal UUID           : {self.uuid}')
        print(f'Animal Species        : {self.species}')
        print(f'Animal LLA            : {self.getLLA()}')
        print(f'Animal Vocal Mean (s) : {self.vocal_interval_mean}')
        print(f'Animal Vocal Std (s)  : {self.vocal_interval_std}')
        
    # motion is modelled as random brownian motion 
    def update_lla(self) -> None:
        
        # get the current lla
        lla = self.getLLA()
        
        # these parameters needs to be tuned
        delta_lat = 0.00001
        delta_lon = 0.00001
        
        # Generate random increments in x and y directions
        dx = delta_lat*np.sqrt(self.clock.step_interval) * np.random.randn(1)
        dy = delta_lon*np.sqrt(self.clock.step_interval) * np.random.randn(1)

        # Compute the cumulative sum of the increments
        lla = ((lla[0] + np.cumsum(dx))[0], (lla[1] + np.cumsum(dy))[0], lla[2])
 
        # update the LLA position
        self.setLLA(lla)
    
    def random_vocalisation(self) -> bool:
        print(f'Random sample for vocalisation')
        vocalisation_flag = False
        
        # get the current sim time
        sim_time = self.clock.get_time()
        
        # calculate how much sim time has elapsed since last vocal
        elapsed = (sim_time - self.last_vocalisation_time).total_seconds() 
        
        # determine if we have hit the threshold for this animal
        if elapsed > self.next_vocal_random_wait:
            
            # TODO: send the vocalisation
            print(f'Vocalisation Sent! Animal {self.uuid} time: {sim_time}')
            vocalisation_flag = True
            
            # calculate when next vocalisation will occur
            self.next_vocal_random_wait = np.random.normal(
                self.vocal_interval_mean,
                self.vocal_interval_std,
                None) 
            
            # track last vocalisation time
            self.last_vocalisation_time = sim_time
            
        return vocalisation_flag   

    def set_random_lla(self) -> None:
        x, y, a = self.randLatLong()
        self.lla = (x, y, a)
        
    def short_uuid(self):
        # Generate a new UUID
        u = uuid.uuid4()

        # Convert the UUID to a 16-byte string
        b = u.bytes

        # Encode the byte string using base64 encoding, and remove padding characters
        encoded = base64.urlsafe_b64encode(b).rstrip(b'=')

        # Decode the byte string to a regular string
        return encoded.decode('utf-8').upper()
        