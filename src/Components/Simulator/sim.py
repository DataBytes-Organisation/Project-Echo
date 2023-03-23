from mic_generator import Map
import os
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from event_generator import Event
from geopy.distance import distance
from math import radians, sin, cos, sqrt, asin, degrees, pi, atan2
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster


class Simulator():
    def __init__(self) -> None:
        self.NODE_THRESHOLD = 0.008
        self.simulated_map = Map()
        self.graph = None
        
    def _connect_mics(self) -> None:
        self.graph = nx.Graph()
        for mic in self.simulated_map.microphones:
            self.graph.add_node(mic)

        for mic1 in self.simulated_map.microphones:
            for mic2 in self.simulated_map.microphones:
                if mic1 != mic2:
                    lat1, long1 = mic1.lat, mic1.long
                    lat2, long2 = mic2.lat, mic2.long
                    _distance = ((lat1 - lat2) ** 2 + (long1 - long2) ** 2) ** 0.5
                    if _distance <= self.NODE_THRESHOLD:
                        self.graph.add_edge(mic1, mic2)

    def _draw_node_graph(self) -> None:
        nx.draw(self.graph, with_labels=False)
        plt.savefig(os.path.join(os.getcwd(),'src/Components/Simulator/mic_node_graph.png'))
        # plt.show()

    def triangulate_event(self, event_lat: float, event_long: float) -> None:
        print(f'\nEvent occured at Lat {event_lat}, and Long {event_long}\n')
        toda_data = []
        distances = []
        for mic in self.simulated_map.microphones:
            mic_distance = distance((event_lat, event_long), (mic.lat, mic.long)).m
            distances.append((mic, mic_distance))

        sorted_mics = sorted(distances, key=lambda x: x[1])

        for mic, dist in sorted_mics[:3]:
            speed_of_sound = 343
            time_diff = dist / speed_of_sound
            mic.T_TS = time_diff
            print(f"Microphone triggered at lat: {mic.lat}, long: {mic.long} with time diff of {time_diff} seconds")
            toda_data.append([mic.lat, mic.long, mic.T_TS])

        if len(toda_data) < 3:
            print("Not enough TDOA data to triangulate event")
            return

        self._triangulate(toda_data, event_lat, event_long)
        # error = distance((event_lat, event_long), (event_lat_est, event_long_est)).m
        # print(f"Triangulation error: {error} meters")

    def _triangulate(self, tdoa_data: list[tuple[float, float, float]], event_lat, event_long) -> tuple[float, float]:
        def get_intersections(latitudes, longitudes, radii):
            R = 6371  # radius of the Earth in km
            intersections = []
            for i in range(len(latitudes)):
                for j in range(i+1, len(latitudes)):
                    lat1, lon1, r1 = latitudes[i], longitudes[i], radii[i]
                    lat2, lon2, r2 = latitudes[j], longitudes[j], radii[j]
                    d = math.acos(math.sin(lat1)*math.sin(lat2) + math.cos(lat1)*math.cos(lat2)*math.cos(lon1-lon2)) * R
                    if d <= r1 + r2:
                        # circles intercept, calculate intersection points
                        a = (r1**2 - r2**2 + d**2) / (2*d)
                        h = math.sqrt(r1**2 - a**2)
                        lat3 = math.asin(math.sin(lat1)*math.cos(h/d) + math.cos(lat1)*math.sin(h/d)*math.cos(lon1-lon2))
                        lon3 = lon1 + math.atan2(math.sin(lon2-lon1)*math.cos(lat2), math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(lon2-lon1))
                        intersections.append((lat3, lon3))
            if len(intersections) > 0:
                print("Intersections found at:")
                for i in range(len(intersections)):
                    print(f"Intersection {i+1}: latitude={intersections[i][0]}, longitude={intersections[i][1]}")
            else:
                print("No intersections found.")

        c = 343 # speed of sound in meters/second
        R = 6371000  # radius of the Earth in meters

        latitudes = [tdoa_data[2][0], tdoa_data[1][0], tdoa_data[0][0]]
        longitudes = [tdoa_data[2][1], tdoa_data[1][1], tdoa_data[0][1]]
        radii = [c * tdoa_data[2][2], c * tdoa_data[1][2], c * tdoa_data[0][2]]
        self.map_intersections(latitudes, longitudes, radii, event_lat, event_long)
        get_intersections(latitudes, longitudes, radii)


    
    def map_intersections(self, latitudes, longitudes, radii, event_lat, event_long):
        marker_cluster = MarkerCluster().add_to(self.simulated_map.folium_map)

        for lat, lon, rad in zip(latitudes, longitudes, radii):
            folium.CircleMarker(
                location=[lat, lon],
                radius=rad,
                color='black',
                fill=False,
                dash_array='10',
                fixed_radius=True
            ).add_to(marker_cluster)

        folium.Marker(location=[event_lat, event_long], icon=folium.Icon(icon="paw", prefix='fa', color="red")).add_to(self.simulated_map.folium_map)

        self.simulated_map.folium_map.save(os.path.join(os.getcwd(),'src/Components/Simulator/map_event_detection.html'))


if __name__ == "__main__":
    ECHO_SIMULATOR = Simulator()
    ECHO_SIMULATOR.simulated_map.print_map()
    ECHO_SIMULATOR._connect_mics()
    ECHO_SIMULATOR._draw_node_graph()
    ECHO_SIMULATOR.triangulate_event(*Event(ECHO_SIMULATOR.simulated_map).get_event_lat_long())

        