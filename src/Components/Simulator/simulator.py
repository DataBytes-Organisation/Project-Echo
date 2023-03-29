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

    def reset_mics(self):
        for mic in self.graph.nodes():
            mic.reset()

    def simulate_event_occuring(self, animal_obect: Animal) -> None:
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
            mic.set_trigger_event_time(animal_obect.get_sound_production_time() + datetime.timedelta(seconds=mic.distance(animal_obect)/self.c))

        
        def find_triggered_mics(graph):
            triggered_mics = []
            for mic in graph.nodes():
                if mic.TRIGGERED:
                    triggered_mics.append(mic)

            triggered_mics.sort(key=lambda x: x.event_timestamp)
            return triggered_mics

        def solve_animal_lla(list_triggered_mics):
            estimate_animal = Animal()
            best_err = 100000000

            for lat in np.arange(self.SIMULATED_OTWAYS_MAP.get_otways_coordinates()[3][0], self.SIMULATED_OTWAYS_MAP.get_otways_coordinates()[1][0], 0.001):
                for lon in np.arange(self.SIMULATED_OTWAYS_MAP.get_otways_coordinates()[4][1], self.SIMULATED_OTWAYS_MAP.get_otways_coordinates()[2][1], 0.001):
                    estimate_animal.setLLA((lat, lon, 10.0))

                    st_est = [mic.distance(estimate_animal) / self.c - np.min([m.distance(estimate_animal) / self.c for m in list_triggered_mics]) for mic in list_triggered_mics]
                    st_obs = [(mic.get_trigger_event_time() - list_triggered_mics[0].get_trigger_event_time()).total_seconds() for mic in list_triggered_mics]

                    err = sum([abs(st_obs[i] - st_est[i]) for i in range(4)])

                    if err < best_err:
                        best_err = err
                        best_lat, best_lon = lat, lon

            return best_lat, best_lon

                        
        # using only the information about sensors and observed trigger times
        triggered_mics_ = find_triggered_mics(self.graph)
        predicted_lla = solve_animal_lla(triggered_mics_)  
        
        if predicted_lla is not None:
            error = distance((animal_obect.getLLA()[0], animal_obect.getLLA()[1]), (predicted_lla[0], predicted_lla[1])).m
            if error < 100:
                print(f'\nEvent occured at Lat, Long pair [{animal_obect.getLLA()[0]}, {animal_obect.getLLA()[1]}] at time: {animal_obect.get_sound_production_time()}\n')
                print(f"\nTriangulation error: {round(error, 2)} meters - predicted lla {predicted_lla}")
                self.map_intersections(predicted_lla, animal_obect.getLLA())
                self.broadcast_message(predicted_lla, triggered_mics_)
            else:
                pass
            self.reset_mics()
        else:
            print('\nInsufficient trilaterate')
    
    def map_intersections(self, predicted_lla, truth_lla) -> None:
        folium.Marker(location=[truth_lla[0], truth_lla[1]], icon=folium.Icon(icon="paw", prefix='fa', color="red"), popup="Animal Truth Location", icon_offset=(0, 0)).add_to(self.SIMULATED_OTWAYS_MAP.folium_map)
        folium.Marker(location=[predicted_lla[0], predicted_lla[1]], icon=folium.Icon(icon="signal", color="green"), popup="Predicted Location", icon_offset=(0, 0)).add_to(self.SIMULATED_OTWAYS_MAP.folium_map)
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

    for _ in range(3):
        ECHO_SIMULATOR.simulate_event_occuring(Animal())

        
        