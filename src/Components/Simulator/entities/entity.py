
#############################################################################
# This class represents any simulated entity sharing common code
#############################################################################

from math import sqrt
from pyproj import Transformer
import math
import random
import numpy as np

class Entity():
    # constructor
    def __init__(self, lla=None) -> None:    
        self.lla = lla
        # Create a Transformer object for LLA to ECEF conversion
        self.to_ecef = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
        self.from_ecef = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)

    # get the earth-centered earth-fixed coordinates of this entity
    def getECEF(self):
        assert self.lla is not None, "The lla attribute must not be None"
        # Convert LLA to ECEF coordinates
        return self.to_ecef.transform(self.lla[1], self.lla[0], self.lla[2])
    
    # set location of entity from ECEF coordinates
    def setECEF(self, ecef):
        lon_lat_alt = self.from_ecef.transform(ecef[0],ecef[1],ecef[2])
        self.lla = (lon_lat_alt[1],lon_lat_alt[0],lon_lat_alt[2])
    
    # get the Latitude longitude altitude location
    def getLLA(self):
        assert self.lla is not None, "The lla attribute must not be None"
        return self.lla
    
    # set the Latitude longitude altitude location
    def setLLA(self, lla):
        self.lla = lla
    
    # get distance in meters to another entity
    def distance(self, other):
        return math.dist(self.getECEF(), other.getECEF())

    # returns a random lat, lon, alt
    def randLatLong(self) -> tuple:
        # Define the four corner points of the diamond
        left_diamond = (-38.78648354661556, 143.5445900890966)
        top_diamond = (-38.77310461001655, 143.5769246453492)
        bottom_diamond = (-38.80412439285561, 143.5796606462629)
        right_diamond = (-38.78299363122898, 143.60726938275553)

        while True:
            # Generate a random latitude value within the vertical bounds of the diamond
            lat = random.uniform(bottom_diamond[0], top_diamond[0])

            # Calculate the left and right longitude bounds based on the current latitude
            left_bound = left_diamond[1] + (left_diamond[1] - right_diamond[1]) * (lat - left_diamond[0]) / (top_diamond[0] - left_diamond[0])
            right_bound = right_diamond[1] + (right_diamond[1] - left_diamond[1]) * (lat - right_diamond[0]) / (bottom_diamond[0] - right_diamond[0])

            # Generate a random longitude value within the calculated bounds
            lon = random.uniform(left_bound, right_bound)

            # Check if the generated lat/lon values fall within the diamond
            if (lon >= left_diamond[1] and lon <= right_diamond[1] and lat >= bottom_diamond[0] and lat <= top_diamond[0]):
                return (lat, lon, 10)
        

    def get_otways_coordinates(self):
        centre_diamond = (-38.78619972614279, 143.5743202660838)
        left_diamond = (-38.78648354661556, 143.5445900890966)
        top_diamond = (-38.77310461001655, 143.5769246453492)
        bottom_diamond = (-38.80412439285561, 143.5796606462629)
        right_diamond = (-38.78299363122898, 143.60726938275553)

        return centre_diamond, top_diamond, right_diamond, bottom_diamond, left_diamond