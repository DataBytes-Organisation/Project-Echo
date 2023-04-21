
#############################################################################
# This class represents a simulated sensor manager for microphones
#############################################################################

import entities.entity
from entities.species import Species
import datetime
from entities.animal import Animal
import entities.microphone
import networkx as nx
import itertools
import numpy as np
from geopy.distance import distance
from scipy.optimize import least_squares


class SensorManager(entities.entity.Entity):
    def __init__(self) -> None:

        super(SensorManager, self).__init__()
        self.MicrophoneObjects = []
        self.GRAPH = nx.Graph()
        self.NODE_THRESHOLD = 0.008
        self.c = 343

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
        for mic in mics_around_event:
            mic.set_trigger_event_time(animal.last_vocalisation_time + datetime.timedelta(seconds=mic.distance(animal)/self.c))

        predicted_lla_1 = self.solve_animal_lla(mics_around_event)
        predicted_lla_2 = self.solve_animal_lla_optimized(mics_around_event)        
        min_error = min(distance((animal.getLLA()[0], animal.getLLA()[1]), (predicted_lla_2[0], predicted_lla_2[1])).m, distance((animal.getLLA()[0], animal.getLLA()[1]), (predicted_lla_1[0], predicted_lla_1[1])).m)

        print(f'Min error: {min_error}')

    def solve_animal_lla(self, list_triggered_mics):
        estimate_animal = Animal(species=Species.dummy_triangulation)
        best_err = np.inf

        for lat in np.arange(self.get_otways_coordinates()[3][0], self.get_otways_coordinates()[1][0], 0.001):
            for lon in np.arange(self.get_otways_coordinates()[4][1], self.get_otways_coordinates()[2][1], 0.001):
                estimate_animal.setLLA((lat, lon, 10.0))

                st_est = [mic.distance(estimate_animal) / self.c - np.min([m.distance(estimate_animal) / self.c for m in list_triggered_mics]) for mic in list_triggered_mics]
                st_obs = [(mic.triggered_sim_clock_time - list_triggered_mics[0].triggered_sim_clock_time).total_seconds() for mic in list_triggered_mics]

                err = sum([abs(st_obs[i] - st_est[i]) for i in range(4)])

                if err < best_err:
                    best_err = err
                    best_lat, best_lon = lat, lon

        return best_lat, best_lon
    
    def func_to_minimize(self, x, list_triggered_mics, c):
        lat, lon, z = x
        estimate_animal = Animal(species=Species.dummy_triangulation)
        estimate_animal.setLLA((lat, lon, z))
        
        st_est = [mic.distance(estimate_animal) / c - np.min([m.distance(estimate_animal) / c for m in list_triggered_mics]) for mic in list_triggered_mics]
        st_obs = [(mic.triggered_sim_clock_time - list_triggered_mics[0].triggered_sim_clock_time).total_seconds() for mic in list_triggered_mics]

        return [st_obs[i] - st_est[i] for i in range(4)]

    def solve_animal_lla_optimized(self, list_triggered_mics):
        lat_start = self.get_otways_coordinates()[3][0]
        lon_start = self.get_otways_coordinates()[4][1]
        z_start = 10.0

        initial_guess = np.array([lat_start, lon_start, z_start])
        res = least_squares(self.func_to_minimize, initial_guess, args=(list_triggered_mics, self.c), method='trf', bounds=([self.get_otways_coordinates()[3][0], self.get_otways_coordinates()[4][1], 0], [self.get_otways_coordinates()[1][0], self.get_otways_coordinates()[2][1], np.inf]))

        best_lat, best_lon, best_z = res.x

        return best_lat, best_lon