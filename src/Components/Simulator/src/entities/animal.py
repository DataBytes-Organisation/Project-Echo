
#############################################################################
# This class represents a simulated animal
#############################################################################

from clock import Clock
import entities.species
import entities.entity
import numpy as np
import base64
import uuid
import numpy as np
import logging
logger1 = logging.getLogger('_sys_logger')

class Animal(entities.entity.Entity):
    
    def __init__(self, species, 
                 lla=None, 
                 vocal_interval_mean=3.0,
                 vocal_interval_std=0.5) -> None:
        
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
        
    def getSpecies(self):
        return self.species
    
    def getUUID(self):
        return self.uuid

    def describe(self) -> None:
        logger1.info(f'Animal UUID           : {self.uuid}')
        logger1.info(f'Animal Species        : {self.species.getName()}')
        logger1.info(f'Animal LLA            : {self.getLLA()}')
        logger1.info(f'Animal Vocal Mean (s) : {self.vocal_interval_mean}')
        logger1.info(f'Animal Vocal Std (s)  : {self.vocal_interval_std}')
        
    # motion is modelled as random brownian motion 
    def update_lla(self) -> None:
        # Get the current position
        lla = self.getLLA()

        # Initialize the motion vector
        dx, dy = 0, 0

        # Define multiple environmental and behavioral parameters
        attraction_factor = 0.001  # Environmental attraction factor
        avoidance_factor = 0.002  # Avoidance factor
        group_behavior_factor = 0.003  # Group behavior factor
        terrain_influence_factor = 0.001  # Terrain influence factor
        weather_influence = 0.001  # Weather influence factor

        # Simulate points of attraction and sources of danger
        attractions = [(1.2, 1.4, 0.3), (2.3, 2.8, 0.2)]
        threats = [(1.5, 1.6, 0.5), (2.1, 2.2, 0.4)]

        # Calculate the impact of environmental factors
        for point in attractions:
            vector_to_point = np.array([point[0] - lla[0], point[1] - lla[1]])
            distance = np.linalg.norm(vector_to_point)
            if distance > 0:
                dx += (attraction_factor * (vector_to_point[0] / distance) * point[2])
                dy += (attraction_factor * (vector_to_point[1] / distance) * point[2])

        for threat in threats:
            vector_to_threat = np.array([threat[0] - lla[0], threat[1] - lla[1]])
            distance = np.linalg.norm(vector_to_threat)
            if distance > 0:
                dx -= (avoidance_factor * (vector_to_threat[0] / distance) * threat[2])
                dy -= (avoidance_factor * (vector_to_threat[1] / distance) * threat[2])

        # Complex effects of weather and seasons
        dx *= (1 + np.random.normal(0, weather_influence))
        dy *= (1 + np.random.normal(0, weather_influence))

        # Influence of terrain
        dx *= (1 + terrain_influence_factor * np.random.rand())
        dy *= (1 + terrain_influence_factor * np.random.rand())

        # Random walking increment
        random_walk_dx = 0.001 * np.sqrt(self.clock.step_interval) * np.random.randn()
        random_walk_dy = 0.001 * np.sqrt(self.clock.step_interval) * np.random.randn()

        # Additional behavioral patterns, such as searching for mates or avoiding other groups
        mating_season = True
        if mating_season:
            potential_mates = [(1.1, 1.2, 0.3)]
            for mate in potential_mates:
                vector_to_mate = np.array([mate[0] - lla[0], mate[1] - lla[1]])
                mate_distance = np.linalg.norm(vector_to_mate)
                if mate_distance > 0:
                    dx += 0.001 * (vector_to_mate[0] / mate_distance) * mate[2]
                    dy += 0.001 * (vector_to_mate[1] / mate_distance) * mate[2]

        # Update position
        new_lla = (lla[0] + dx + random_walk_dx, lla[1] + dy + random_walk_dy, lla[2])

        # Check if within boundaries and adjust if necessary
        if not self.is_within_boundaries(new_lla):
            new_lla = self.adjust_to_boundaries(new_lla)

        # Set the new position
        self.setLLA(new_lla)

    def is_within_boundaries(self, lla):
        lat, lon, _ = lla
        if (lon >= self.left_diamond[1] and lon <= self.right_diamond[1] and 
            lat >= self.bottom_diamond[0] and lat <= self.top_diamond[0]):
            return True
        return False

    def adjust_to_boundaries(self, lla):
        lat, lon, alt = lla
        if lat > self.top_diamond[0]:
            lat = self.top_diamond[0]
        elif lat < self.bottom_diamond[0]:
            lat = self.bottom_diamond[0]

        if lon > self.right_diamond[1]:
            lon = self.right_diamond[1]
        elif lon < self.left_diamond[1]:
            lon = self.left_diamond[1]

        return (lat, lon, alt)

    
    def random_vocalisation(self) -> bool:
        logger1.info(f'Random sample for vocalisation')
        vocalisation_flag = False
        
        # get the current sim time
        sim_time = self.clock.get_time()
        
        # calculate how much sim time has elapsed since last vocal
        elapsed = (sim_time - self.last_vocalisation_time).total_seconds() 
        
        # determine if we have hit the threshold for this animal
        if elapsed > self.next_vocal_random_wait:
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
        