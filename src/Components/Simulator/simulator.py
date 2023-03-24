import json
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

    def triangulate_event(self, lla) -> None:
        print(f'\nEvent occured at Lat {lla[0]}, and Long {lla[1]}\n')
        event_lat = lla[0]
        event_long = lla[1]

        toda_data = []
        distances = []

        for mic in self.SIMULATED_OTWAYS_MAP.microphones:
            mic_distance = distance((event_lat, event_long), (mic.getLLA()[0], mic.getLLA()[1])).m
            distances.append((mic, mic_distance))

        sorted_mics = sorted(distances, key=lambda x: x[1])

        for mic, dist in sorted_mics[:3]:
            c = 343
            time_diff = dist / c
            mic.set_time_delay(time_diff)
            print(f"Microphone triggered at lat: {mic.getLLA()[0]}, long: {mic.getLLA()[1]} with time diff of {time_diff} seconds")
            toda_data.append(mic)

        if len(toda_data) < 3:
            print("Not enough TDOA data to triangulate event")
            return
        
        # Show mic souond radius on map as well as event location and output map
        marker_cluster = MarkerCluster().add_to(self.SIMULATED_OTWAYS_MAP.folium_map)

        for mic in toda_data:
            folium.CircleMarker(
                location=[mic.getLLA()[0], mic.getLLA()[1]],
                radius=mic.get_time_delay()*c,
                color='black',
                fill=False,
                dash_array='10'
            ).add_to(marker_cluster)

        folium.Marker(location=[event_lat, event_long], icon=folium.Icon(icon="paw", prefix='fa', color="red"), popup="Event Location").add_to(self.SIMULATED_OTWAYS_MAP.folium_map)
        self.SIMULATED_OTWAYS_MAP.folium_map.save(os.path.join(os.getcwd(),'src/Components/Simulator/outputs/map_event_detection.html'))
        
        predicted_lla = self._triangulate(toda_data)
        if predicted_lla is not None:
            error = distance((event_lat, event_long), (predicted_lla[0], predicted_lla[1])).m
            print(f"\nTriangulation error: {round(error, 2)} meters")
            self.map_intersections(predicted_lla)
            self.broadcast_message(predicted_lla, toda_data)
        else:
            print('\nInsufficient trilaterate')

    def _triangulate(self, tdoa_data) -> tuple[float, float]:
        def trilaterate(p1, r1, p2, r2, p3, r3):
            # Calculate relative positions of point 2 and point 3
            ex = (p2 - p1) / np.linalg.norm(p2 - p1)
            i = np.dot(ex, p3 - p1)
            ey = (p3 - p1 - i * ex) / np.linalg.norm(p3 - p1 - i * ex)
            ez = np.cross(ex, ey)

            # Calculate the distances
            d = np.linalg.norm(p2 - p1)
            j = np.dot(ey, p3 - p1)

            # Calculate the position of the intersection point
            x = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
            y = (r1 ** 2 - r3 ** 2 + i ** 2 + j ** 2) / (2 * j) - (i / j) * x

            # Calculate the z-coordinate, if possible
            z_squared = r1 ** 2 - x ** 2 - y ** 2
            if z_squared <= 0:
                return None  # No intersection
            z = np.sqrt(z_squared)

            # Calculate the coordinates
            p = p1 + x * ex + y * ey + z * ez
            return p
        
        c = 343 # Speed of sound in m/s through air

        _p_ = []
        _r_ = []
        for mic in tdoa_data:
            _p_.append(np.array(mic.getECEF()))
            _r_.append((mic.get_time_delay() * c) * 1.001) # add weight to radius

        intersection = trilaterate(_p_[0], _r_[0], _p_[1], _r_[1], _p_[2], _r_[2])

        if intersection is not None:
            predicted_animal = Animal()
            predicted_animal.setECEF(intersection)
            return predicted_animal.getLLA()
    
    def map_intersections(self, predicted_lla) -> None:
        folium.Marker(location=[predicted_lla[0] + 0.0001, predicted_lla[1] + 0.0001], icon=folium.Icon(icon="signal-stream", color="green"), popup="Predicted Location", icon_offset=(0, 0)).add_to(self.SIMULATED_OTWAYS_MAP.folium_map)
        self.SIMULATED_OTWAYS_MAP.folium_map.save(os.path.join(os.getcwd(),'src/Components/Simulator/outputs/map_event_detection.html'))
    
    def broadcast_message(self, predicted_lla, toda_data) -> None:
        _tmp_json_data = {
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

    ECHO_SIMULATOR.triangulate_event(Animal().getLLA())

        
        