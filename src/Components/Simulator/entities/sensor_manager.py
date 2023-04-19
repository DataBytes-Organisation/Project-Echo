
#############################################################################
# This class represents a simulated sensor manager for microphones
#############################################################################

import entities.entity
import datetime
import entities.microphone
import networkx as nx

class SensorManager(entities.entity.Entity):
    def __init__(self) -> None:
        self.MicrophoneObjects = []
        self.GRAPH = nx.Graph()
        self.NODE_THRESHOLD = 0.008

    def add_sensor_object(self, MicObj: entities.microphone) -> None:
        self.MicrophoneObjects.append(MicObj)
    
    def generate_sensor_graph(self) -> None:
        for mic in self.MicrophoneObjects:
            self.GRAPH.add_node(mic)

        for mic1 in self.MicrophoneObjects:
            for mic2 in self.MicrophoneObjects:
                if mic1 != mic2:
                    lat1, long1 = mic1.getLLA()[0], mic1.getLLA()[1]
                    lat2, long2 = mic2.getLLA()[0], mic2.getLLA()[1]
                    _distance = ((lat1 - lat2) ** 2 + (long1 - long2) ** 2) ** 0.5
                    if _distance <= self.NODE_THRESHOLD:
                        self.GRAPH.add_edge(mic1, mic2)