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

        print(self._triangulate(toda_data, event_lat, event_long))
        # error = distance((event_lat, event_long), (event_lat_est, event_long_est)).m
        # print(f"Triangulation error: {error} meters")

    def _triangulate(self, tdoa_data: list[tuple[float, float, float]], event_lat, event_long) -> tuple[float, float]:
        def circle_intersection(lat1, lon1, r1, lat2, lon2, r2):
            # Convert latitudes and longitudes to radians
            lat1, lon1, lat2, lon2 = map(lambda x: x * pi / 180, [lat1, lon1, lat2, lon2])
            # Calculate distance between circle centers
            d_lat = lat2 - lat1
            d_lon = lon2 - lon1
            a = sin(d_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(d_lon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            # Calculate intersection points (if any)
            if abs(r1 - r2) <= c * r1 <= r1 + r2:
                # One intersection point
                d = r1 ** 2 - (r1 ** 2 - r2 ** 2 + c ** 2 * r1 ** 2) / (2 * c)
                if d < 0:
                    return [] # no intersection points
                h = sqrt(d)
                lat = lat1 + d_lat * d / c
                lon = lon1 + d_lon * d / c
                lat1_int = lat + h * (lon2 - lon1) / c
                lon1_int = lon - h * (lat2 - lat1) / c
                return [(lat1_int * 180 / pi, lon1_int * 180 / pi)]
            elif c * r1 < abs(r1 - r2) or c * r1 > r1 + r2:
                # No intersection points
                return []
            else:
                # Two intersection points
                d = (r1 ** 2 - r2 ** 2 + c ** 2 * r1 ** 2) / (2 * c)
                if abs(r1 - r2) <= c * r1 <= r1 + r2 and d < 0:
                    return [] # no intersection points
                h = sqrt(r1 ** 2 - d ** 2)
                lat = lat1 + d_lat * d / c
                lon = lon1 + d_lon * d / c
                lat1_int = lat + h * (lon2 - lon1) / c
                lon1_int = lon - h * (lat2 - lat1) / c
                lat2_int = lat - h * (lon2 - lon1) / c
                lon2_int = lon + h * (lat2 - lat1) / c
                if lat1_int == lat2_int and lon1_int == lon2_int:
                    return [(lat1_int * 180 / pi, lon1_int * 180 / pi)]
                else:
                    return [(lat1_int * 180 / pi, lon1_int * 180 / pi), (lat2_int * 180 / pi, lon2_int * 180 / pi)]

            
        c = 343 # speed of sound in m/s

        latitudes = [tdoa_data[2][0], tdoa_data[1][0], tdoa_data[0][0]]
        longitudes = [tdoa_data[2][1], tdoa_data[1][1], tdoa_data[0][1]]
        radii = [(c * tdoa_data[2][2]), (c * tdoa_data[1][2]), (c * tdoa_data[0][2])]
        self.map_intersections(latitudes, longitudes, radii, event_lat, event_long)

        # Test intersection of first two circles
        intersection_points = circle_intersection(latitudes[0], longitudes[0], radii[0], latitudes[1], longitudes[1], radii[1])
        print(intersection_points)


    
    def map_intersections(self, latitudes, longitudes, radii, event_lat, event_long):
        marker_cluster = MarkerCluster().add_to(self.simulated_map.folium_map)

        for lat, lon, rad in zip(latitudes, longitudes, radii):
            folium.CircleMarker(
                location=[lat, lon],
                radius=rad,
                color='black',
                fill=False,
                dash_array='10'
            ).add_to(marker_cluster)

        folium.Marker(location=[event_lat, event_long], icon=folium.Icon(icon="paw", prefix='fa', color="red")).add_to(self.simulated_map.folium_map)

        self.simulated_map.folium_map.save(os.path.join(os.getcwd(),'src/Components/Simulator/map_event_detection.html'))


if __name__ == "__main__":
    ECHO_SIMULATOR = Simulator()
    ECHO_SIMULATOR.simulated_map.print_map()
    ECHO_SIMULATOR._connect_mics()
    ECHO_SIMULATOR._draw_node_graph()
    ECHO_SIMULATOR.triangulate_event(*Event(ECHO_SIMULATOR.simulated_map).get_event_lat_long())

        