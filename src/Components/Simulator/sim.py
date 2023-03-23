from mic_generator import Map
import os
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from event_generator import Event
from geopy.distance import distance
from math import radians, sin, cos, sqrt, atan2, degrees
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

        event_lat_est, event_long_est = self._triangulate(toda_data, event_lat, event_long)
        error = distance((event_lat, event_long), (event_lat_est, event_long_est)).m
        print(f"Triangulation error: {error} meters")

    def _triangulate(self, tdoa_data: list[tuple[float, float, float]], event_lat, event_long) -> tuple[float, float]:
        #assuming elevation = 0
        earthR = 6371
        c = 343*2 # speed of sound in meters/second

        LatA = tdoa_data[2][0]
        LonA = tdoa_data[2][1]
        DistA = c * tdoa_data[2][2]

        LatB = tdoa_data[1][0]
        LonB = tdoa_data[1][1]
        DistB = c * tdoa_data[1][2]

        LatC = tdoa_data[0][0]
        LonC = tdoa_data[0][1]
        DistC = c * tdoa_data[0][2]

        latitudes = [LatA, LatB, LatC]
        longitudes = [LonA, LonB, LonC]
        radii = [DistA, DistB, DistC]

        self.map_intersections(latitudes, longitudes, radii, event_lat, event_long)
        input()

        #using authalic sphere
        #if using an ellipsoid this step is slightly different
        #Convert geodetic Lat/Long to ECEF xyz
        #   1. Convert Lat/Long to radians
        #   2. Convert Lat/Long(radians) to ECEF
        xA = earthR *(math.cos(math.radians(LatA)) * math.cos(math.radians(LonA)))
        yA = earthR *(math.cos(math.radians(LatA)) * math.sin(math.radians(LonA)))
        zA = earthR *(math.sin(math.radians(LatA)))

        xB = earthR *(math.cos(math.radians(LatB)) * math.cos(math.radians(LonB)))
        yB = earthR *(math.cos(math.radians(LatB)) * math.sin(math.radians(LonB)))
        zB = earthR *(math.sin(math.radians(LatB)))

        xC = earthR *(math.cos(math.radians(LatC)) * math.cos(math.radians(LonC)))
        yC = earthR *(math.cos(math.radians(LatC)) * math.sin(math.radians(LonC)))
        zC = earthR *(math.sin(math.radians(LatC)))

        P1 = np.array([xA, yA, zA])
        P2 = np.array([xB, yB, zB])
        P3 = np.array([xC, yC, zC])

        #from wikipedia
        #transform to get circle 1 at origin
        #transform to get circle 2 on x axis
        ex = (P2 - P1)/(np.linalg.norm(P2 - P1))
        i = np.dot(ex, P3 - P1)
        ey = (P3 - P1 - i*ex)/(np.linalg.norm(P3 - P1 - i*ex))
        ez = np.cross(ex,ey)
        d = np.linalg.norm(P2 - P1)
        j = np.dot(ey, P3 - P1)

        #from wikipedia
        #plug and chug using above values
        x = (pow(DistA,2) - pow(DistB,2) + pow(d,2))/(2*d)
        y = ((pow(DistA,2) - pow(DistC,2) + pow(i,2) + pow(j,2))/(2*j)) - ((i/j)*x)

        # only one case shown here
        z = np.sqrt(pow(DistA,2) - pow(x,2) - pow(y,2))

        #triPt is an array with ECEF x,y,z of trilateration point
        triPt = P1 + x*ex + y*ey + z*ez

        #convert back to lat/long from ECEF
        #convert to degrees and return
        return math.degrees(math.asin(triPt[2] / earthR)), math.degrees(math.atan2(triPt[1],triPt[0]))
    
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