import random
import os
import datetime
from datetime import datetime as dt
from geopy.distance import distance
from shapely.geometry import Polygon, Point
import folium
import numpy as np

class Map:
    class Microphone:
        def __init__(self, lat, long) -> None:
            self.lat = lat
            self.long = long
            self.T_TS = None

    def __init__(self):
        self.otways_size = (20, 15)
        self.square_size = 3
        self.folium_map = None

        self.microphones = []
        # -38.790144470951354, 143.52798948464255
        # -38.79177529796978, 143.5965127875795
        self.otways_coordinates = [-38.790144470951354, -38.79177529796978, 143.52798948464255, 143.5965127875795]

        self.main()

    def add_microphone(self, microphone):
        self.microphones.append(microphone)    

    def generate_microphones(self, num_mics) -> list:
        mic_list = []
        
        square_lat_half = self.square_size/2*1.1/111.0
        square_long_half = self.square_size/2*1.1/111.0/np.cos(np.deg2rad((self.otways_coordinates[0]+self.otways_coordinates[1])/2))
        
        center_lat = (self.otways_coordinates[0]+self.otways_coordinates[1])/2
        center_long = (self.otways_coordinates[2]+self.otways_coordinates[3])/2
        
        square_boundary = Polygon([(center_lat+square_lat_half, center_long+square_long_half),
                                (center_lat+square_lat_half, center_long-square_long_half),
                                (center_lat-square_lat_half, center_long-square_long_half),
                                (center_lat-square_lat_half, center_long+square_long_half)])
        while len(mic_list) < num_mics:
            point = Point(random.uniform(center_lat-square_lat_half, center_lat+square_lat_half),
                        random.uniform(center_long-square_long_half, center_long+square_long_half))
            
            if not square_boundary.contains(point):
                continue
            
            too_close = False
            for mic in mic_list:
                if distance((point.x, point.y), (mic.lat, mic.long)).m < 100:
                    too_close = True
                    break
            if not too_close:
                new_mic = self.Microphone(point.x, point.y)
                self.add_microphone(new_mic)
                mic_list.append(new_mic)
        return mic_list
    
    def main(self):
        self.generate_microphones(100)

    def print_map(self):
        map_center = [(self.otways_coordinates[0] + self.otways_coordinates[1])/2, (self.otways_coordinates[2] + self.otways_coordinates[3])/2]
        m = folium.Map(location=map_center, zoom_start=15)

        for mic in self.microphones:
            folium.Marker(location=[mic.lat, mic.long], icon=folium.Icon(icon="microphone", prefix='fa', color="orange")).add_to(m)

        m.save(os.path.join(os.getcwd(),'src/Components/Simulator/outputs/map.html'))
        self.folium_map = m

if __name__ == "__main__":
    print("Meant to be imported. Not run.")