
#############################################################################
# This class represents any simulated entity sharing common code
#############################################################################

from pyproj import Transformer

class Entity():
    def __init__(self) -> None:
        
        # Set a valid default position
        self.lla = (-38.0,143.0)
        
        # Create a Transformer object for LLA to ECEF conversion
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)

    # get the earth-centered earth-fixed coordinates of this entity
    def getECEF(self):
        # Convert LLA to ECEF coordinates
        return self.transformer.transform(self.lla[1], self.lla[0], self.lla[2])
    
    # get the Latitude longitude altitude location
    def getLLA(self):
        return self.lla
    
    # get distance in meters to another entity
    def distance(self, other):
        
        
    
    