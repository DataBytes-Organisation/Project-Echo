
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

        predicted_lla = self.solve_animal_lla_optimized(mics_around_event, animal)        
        min_error = distance((animal.getLLA()[0], animal.getLLA()[1]), (predicted_lla[0], predicted_lla[1])).m

        for mic in mics_around_event:
            mic.reset_trigger_event_time()

        print(f'Min triangulation error: {min_error}\n')
        return predicted_lla
    
    def func_to_minimize(self, x, list_triggered_mics, c):
        lat, lon, z = x
        estimate_animal = Animal(species=Species("dummy_triangulation"))
        estimate_animal.setLLA((lat, lon, z))
        
        st_est = [mic.distance(estimate_animal) / c - np.min([m.distance(estimate_animal) / c for m in list_triggered_mics]) for mic in list_triggered_mics]
        st_obs = [(mic.triggered_sim_clock_time - list_triggered_mics[0].triggered_sim_clock_time).total_seconds() for mic in list_triggered_mics]

        return [st_obs[i] - st_est[i] for i in range(4)]

    def solve_animal_lla_optimized(self, list_triggered_mics, truth_animal):
        z_start = 10.0

        # Improved initial guess using the average of the microphones' positions
        avg_lat = np.mean([mic.getLLA()[0] for mic in list_triggered_mics])
        avg_lon = np.mean([mic.getLLA()[1] for mic in list_triggered_mics])
        improved_initial_guess = np.array([avg_lat, avg_lon, z_start])

        max_retries = 10
        retry_count = 0
        large_error_threshold = 50  # acceptable error threshold in meters

        while retry_count < max_retries:
            res = least_squares(
                self.func_to_minimize,
                improved_initial_guess,
                args=(list_triggered_mics, self.c),
                method="trf",
                bounds=(
                    [self.get_otways_coordinates()[3][0], self.get_otways_coordinates()[4][1], 0],
                    [self.get_otways_coordinates()[1][0], self.get_otways_coordinates()[2][1], np.inf],
                ),
            )

            best_lat, best_lon, best_z = res.x
            predicted_lla = (best_lat, best_lon)

            error = distance((truth_animal.getLLA()[0], truth_animal.getLLA()[1]), predicted_lla).m

            if error < large_error_threshold:
                break
            else:
                # Update the initial guess to try again
                low_bounds = [
                    max(self.get_otways_coordinates()[3][0], best_lat - 0.01),
                    max(self.get_otways_coordinates()[4][1], best_lon - 0.01),
                    0,
                ]
                high_bounds = [
                    min(self.get_otways_coordinates()[1][0], best_lat + 0.01),
                    min(self.get_otways_coordinates()[2][1], best_lon + 0.01),
                    np.inf,
                ]

                improved_initial_guess = np.random.uniform(low=low_bounds, high=high_bounds)

                retry_count += 1

        return best_lat, best_lon, best_z