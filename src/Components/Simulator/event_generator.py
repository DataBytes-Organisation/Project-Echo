import random
import os
import datetime
from datetime import datetime as dt
from geopy.distance import distance
from shapely.geometry import Polygon, Point
import folium
import numpy as np

class Event:
    def __init__(self, in_sim_map) -> None:
        self.timestamp = dt.now()
        self.SIM_MAP = in_sim_map
        self.SECRET_LAT, self.SECRET_LONG = self.generate_secret_location()

    def generate_secret_location(self):
        square_lat_half = self.SIM_MAP.square_size / 2 * 1.1 / 111.0
        square_long_half = self.SIM_MAP.square_size / 2 * 1.1 / 111.0 / np.cos(np.deg2rad((self.SIM_MAP.otways_coordinates[0] + self.SIM_MAP.otways_coordinates[1]) / 2))
        center_lat = (self.SIM_MAP.otways_coordinates[0] + self.SIM_MAP.otways_coordinates[1]) / 2
        center_long = (self.SIM_MAP.otways_coordinates[2] + self.SIM_MAP.otways_coordinates[3]) / 2

        square_boundary = Polygon([(center_lat + square_lat_half, center_long + square_long_half),
                                   (center_lat + square_lat_half, center_long - square_long_half),
                                   (center_lat - square_lat_half, center_long - square_long_half),
                                   (center_lat - square_lat_half, center_long + square_long_half)])

        while True:
            point = Point(random.uniform(center_lat - square_lat_half, center_lat + square_lat_half),
                          random.uniform(center_long - square_long_half, center_long + square_long_half))
            if square_boundary.contains(point):
                break

        return point.x, point.y
    
    def get_event_lat_long(self) -> list[float]:
        return self.SECRET_LAT, self.SECRET_LONG


if __name__ == "__main__":
    print("Meant to be imported. Not run.")