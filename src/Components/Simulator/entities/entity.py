
#############################################################################
# This class represents any simulated entity sharing common code
#############################################################################

import math
from pyproj import Transformer
from shapely.geometry import Polygon, Point
import random
import numpy as np

class Entity():
    # constructor
    def __init__(self, lla) -> None:
        
        # Set start position (latitude, longitude, altitude)
        self.lla = lla
        
        # Create a Transformer object for LLA to ECEF conversion
        self.to_ecef = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
        self.from_ecef = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)

    # get the earth-centered earth-fixed coordinates of this entity
    def getECEF(self):
        # Convert LLA to ECEF coordinates
        return self.to_ecef.transform(self.lla[1], self.lla[0], self.lla[2])
    
    # set location of entity from ECEF coordinates
    def setECEF(self, ecef):
        lon_lat_alt = self.from_ecef.transform(ecef[0],ecef[1],ecef[2])
        self.lla = (lon_lat_alt[1],lon_lat_alt[0],lon_lat_alt[2])
    
    # get the Latitude longitude altitude location
    def getLLA(self):
        return self.lla
    
    # get distance in meters to another entity
    def distance(self, other):
        return math.dist(self.getECEF(), other.getECEF())
    
    def randLatLong(self) -> list[float]:
        """
        This function will generate and return a random Lat and Long value within the boundaries
        of the Otways National Forest square size of 300m x 300m
        """
        square_size = 3
        otways_coordinates = [-38.790144470951354, -38.79177529796978, 143.52798948464255, 143.5965127875795]

        square_lat_half = square_size / 2 * 1.1 / 111.0
        square_long_half = square_size / 2 * 1.1 / 111.0 / np.cos(np.deg2rad((otways_coordinates[0] + otways_coordinates[1]) / 2))
        center_lat = (otways_coordinates[0] + otways_coordinates[1]) / 2
        center_long = (otways_coordinates[2] + otways_coordinates[3]) / 2

        square_boundary = Polygon([(center_lat + square_lat_half, center_long + square_long_half),
                                   (center_lat + square_lat_half, center_long - square_long_half),
                                   (center_lat - square_lat_half, center_long - square_long_half),
                                   (center_lat - square_lat_half, center_long + square_long_half)])

        while True:
            point = Point(random.uniform(center_lat - square_lat_half, center_lat + square_lat_half),
                          random.uniform(center_long - square_long_half, center_long + square_long_half))
            if square_boundary.contains(point):
                break

        return [point.x, point.y]

    def get_otways_coordinates(self) -> list[float]:
        return [-38.790144470951354, -38.79177529796978, 143.52798948464255, 143.5965127875795]
        