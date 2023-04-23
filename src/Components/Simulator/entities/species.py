
#############################################################################
# This class represents a simulated species
#############################################################################

class Species():
    # constructor
    def __init__(self, name) -> None:    
        self.name = name
    
    # get the name of the species
    def getName(self):
        return self.name
    
    # TODO: there may be other models here on how this species moves    
