
#############################################################################
# This class represents any simulated entity sharing common code
#############################################################################

import math
from pyproj import Transformer

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
        