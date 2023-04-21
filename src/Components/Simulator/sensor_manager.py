
#############################################################################
# This class represents a simulated sensor manager for microphones
#############################################################################

import entities.entity
import datetime
import entities.microphone
import networkx as nx
import itertools

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

        for mic1, mic2 in itertools.combinations(self.MicrophoneObjects, 2):
            lat1, long1 = mic1.getLLA()[0], mic1.getLLA()[1]
            lat2, long2 = mic2.getLLA()[0], mic2.getLLA()[1]
            _distance = ((lat1 - lat2) ** 2 + (long1 - long2) ** 2) ** 0.5
            if _distance <= self.NODE_THRESHOLD:
                self.GRAPH.add_edge(mic1, mic2)

    def find_closest_mics(self, event_lat, event_long, n):
        mics_distances = [(mic, ((mic.getLLA()[0] - event_lat) ** 2 + (mic.getLLA()[1] - event_long) ** 2) ** 0.5) for mic in self.GRAPH.nodes()]
        mics_distances.sort(key=lambda x: x[1])
        return [mic for mic, _ in mics_distances[:n]]

    def vocalisation(self, animal) -> None:
        mics_around_event = self.find_closest_mics(animal.getLLA()[0], animal.getLLA()[1], 4)
        # for mic in mics_around_event:
        #     mic.set_trigger_event_time(animal.get_sound_production_time() + datetime.timedelta(seconds=mic.distance(animal)/self.c))