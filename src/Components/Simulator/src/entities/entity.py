
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
        
        # Initialise coordinates for otways
        self.centre_diamond = (-38.78628264411816, 143.55917569624032)
        self.left_diamond = (-38.794128852897394, 143.50546901793675)
        self.top_diamond = (-38.76507546897413, 143.576881064484)
        self.bottom_diamond = (-38.808297607205255, 143.56535698715805)
        self.right_diamond = (-38.792808699797234, 143.59295653156977)
        self.altitude = 10.0

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
        while True:
            # Generate a random latitude value within the vertical bounds of the diamond
            lat = random.uniform(self.bottom_diamond[0], self.top_diamond[0])

            # Calculate the left and right longitude bounds based on the current latitude
            left_bound = self.left_diamond[1] + (self.left_diamond[1] - self.right_diamond[1]) * (lat - self.left_diamond[0]) / (self.top_diamond[0] - self.left_diamond[0])
            right_bound = self.right_diamond[1] + (self.right_diamond[1] - self.left_diamond[1]) * (lat - self.right_diamond[0]) / (self.bottom_diamond[0] - self.right_diamond[0])

            # Generate a random longitude value within the calculated bounds
            lon = random.uniform(left_bound, right_bound)

            # Check if the generated lat/lon values fall within the diamond
            if (lon >= self.left_diamond[1] and lon <= self.right_diamond[1] and lat >= self.bottom_diamond[0] and lat <= self.top_diamond[0]):
                return (lat, lon, self.altitude)
        

    def get_otways_coordinates(self):
        return self.centre_diamond, self.top_diamond, self.right_diamond, self.bottom_diamond, self.left_diamond