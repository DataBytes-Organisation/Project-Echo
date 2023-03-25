import json
from datetime import datetime as dt
import datetime
import networkx as nx
import uuid
import os
import networkx as nx
import matplotlib.pyplot as plt
from geopy.distance import distance
from math import radians, sin, cos, sqrt, asin, degrees, pi, atan2
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from simulate_otways import Otways
from entities.animal import Animal 
import numpy as np


class Simulator():
    def __init__(self) -> None:
        self.NODE_THRESHOLD = 0.008
        self.SIMULATED_OTWAYS_MAP = Otways()
        self.graph = None
        self.c = 343 # Speed of sound in m/s
        
    def _connect_mics(self) -> None:
        self.graph = nx.Graph()
        for mic in self.SIMULATED_OTWAYS_MAP.microphones:
            self.graph.add_node(mic)

        for mic1 in self.SIMULATED_OTWAYS_MAP.microphones:
            for mic2 in self.SIMULATED_OTWAYS_MAP.microphones:
                if mic1 != mic2:
                    lat1, long1 = mic1.getLLA()[0], mic1.getLLA()[1]
                    lat2, long2 = mic2.getLLA()[0], mic2.getLLA()[1]
                    _distance = ((lat1 - lat2) ** 2 + (long1 - long2) ** 2) ** 0.5
                    if _distance <= self.NODE_THRESHOLD:
                        self.graph.add_edge(mic1, mic2)

    def _draw_node_graph(self, plt_noodes: bool = False) -> None:
        nx.draw(self.graph, with_labels=False)
        plt.savefig(os.path.join(os.getcwd(),'src/Components/Simulator/outputs/mic_node_graph.png'))
        if plt_noodes: plt.show()

    def simulate_event_occuring(self, animal_obect: Animal) -> None:
        print(f'\nEvent occured at Lat, Long pair [{animal_obect.getLLA()[0]}, {animal_obect.getLLA()[1]}] at time: {animal_obect.get_sound_production_time()}\n')

        def find_closest_mics(graph, event_lat, event_long, n):
            mics_distances = []
            for mic in graph.nodes():
                mic_lat, mic_long = mic.getLLA()[0], mic.getLLA()[1]
                distance = ((mic_lat - event_lat) ** 2 + (mic_long - event_long) ** 2) ** 0.5
                mics_distances.append((mic, distance))
            mics_distances.sort(key=lambda x: x[1])
            return [mic for mic, _ in mics_distances[:n]]

        mics_around_event = find_closest_mics(self.graph, animal_obect.getLLA()[0], animal_obect.getLLA()[1], 4)
        for mic in mics_around_event:
            mic.set_trigger_event_time(animal_obect.get_sound_production_time() - datetime.timedelta(seconds=mic.distance(animal_obect)/self.c))

        
        def find_triggered_mics(graph):
            triggered_mics = []
            for mic in graph.nodes():
                if mic.TRIGGERED:
                    triggered_mics.append(mic)
            return triggered_mics

        for mic in find_triggered_mics(self.graph):
            print(mic.get_trigger_event_time())
        input()
        
        # marker_cluster = MarkerCluster().add_to(self.SIMULATED_OTWAYS_MAP.folium_map)

        # for mic in toda_data:
        #     folium.CircleMarker(
        #         location=[mic.getLLA()[0], mic.getLLA()[1]],
        #         radius=mic.get_time_delay()*c,
        #         color='black',
        #         fill=False,
        #         dash_array='10'
        #     ).add_to(marker_cluster)

        # folium.Marker(location=[event_lat, event_long], icon=folium.Icon(icon="paw", prefix='fa', color="red"), popup="Event Location").add_to(self.SIMULATED_OTWAYS_MAP.folium_map)
        # self.SIMULATED_OTWAYS_MAP.folium_map.save(os.path.join(os.getcwd(),'src/Components/Simulator/outputs/map_event_detection.html'))
        
        # predicted_lla = self._triangulate(toda_data)
        # if predicted_lla is not None:
        #     error = distance((event_lat, event_long), (predicted_lla[0], predicted_lla[1])).m
        #     print(f"\nTriangulation error: {round(error, 2)} meters")
        #     self.map_intersections(predicted_lla)
        #     self.broadcast_message(predicted_lla, toda_data)
        # else:
        #     print('\nInsufficient trilaterate')
    
    def map_intersections(self, predicted_lla) -> None:
        folium.Marker(location=[predicted_lla[0] + 0.0001, predicted_lla[1] + 0.0001], icon=folium.Icon(icon="signal-stream", color="green"), popup="Predicted Location", icon_offset=(0, 0)).add_to(self.SIMULATED_OTWAYS_MAP.folium_map)
        self.SIMULATED_OTWAYS_MAP.folium_map.save(os.path.join(os.getcwd(),'src/Components/Simulator/outputs/map_event_detection.html'))
    
    def broadcast_message(self, predicted_lla, toda_data) -> None:
        _tmp_json_data = {
            "event_ID" : str(uuid.uuid1()),
            "mic_name" : toda_data[0].name,
            "mic_uuid" : toda_data[0].unique_identifier,
            "event_coordinates" : predicted_lla,
            "audio_sample" : "audio_sample.mp3"
        }

        print(json.dumps(_tmp_json_data))


if __name__ == "__main__":
    ECHO_SIMULATOR = Simulator()
    ECHO_SIMULATOR.SIMULATED_OTWAYS_MAP.print_map()

    ECHO_SIMULATOR._connect_mics()
    ECHO_SIMULATOR._draw_node_graph()

    ECHO_SIMULATOR.simulate_event_occuring(Animal())

        
        