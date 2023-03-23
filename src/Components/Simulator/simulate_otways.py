import random
import os
import datetime
from datetime import datetime as dt
from geopy.distance import distance
from shapely.geometry import Polygon, Point
import folium
import entities.entity
import numpy as np
from entities.microphone import MicrophoneStation

class Otways(entities.entity.Entity):
    def __init__(self):
        self.folium_map = None

        self.microphones = []
        super(Otways, self).__init__(lla=(0, 0, 10))
        self.main()

    def add_microphone(self, microphone):
        self.microphones.append(microphone)    

    def generate_microphones(self, num_mics) -> list:
        mic_list = []
        
        while len(mic_list) < num_mics:
            x, y = self.randLatLong()
            
            too_close = False
            for mic in mic_list:
                if distance((x, y), (mic.getLLA()[0], mic.getLLA()[1])).m < 100:
                    too_close = True
                    break
            if not too_close:
                new_mic = MicrophoneStation(lla=(x, y, 10.0))
                self.add_microphone(new_mic)
                mic_list.append(new_mic)
        return mic_list
    
    def main(self):
        self.generate_microphones(100)

    def print_map(self):
        map_center = [(self.get_otways_coordinates()[0] + self.get_otways_coordinates()[1])/2, (self.get_otways_coordinates()[2] + self.get_otways_coordinates()[3])/2]
        m = folium.Map(location=map_center, zoom_start=15)

        for mic in self.microphones:
            folium.Marker(location=[mic.getLLA()[0], mic.getLLA()[1]], icon=folium.Icon(icon="microphone", prefix='fa', color="orange")).add_to(m)

        m.save(os.path.join(os.getcwd(),'src/Components/Simulator/outputs/map.html'))
        self.folium_map = m

if __name__ == "__main__":
    print("Meant to be imported. Not run.")