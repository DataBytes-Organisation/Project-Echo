#############################################################################
# Implementation for Simulated Sensor Manager
#############################################################################

import entities.entity
from entities.species import Species
from entities.animal import Animal
import entities.microphone
import networkx as nx
from geopy.distance import geodesic
from scipy.optimize import minimize
import numpy as np
import datetime
import logging
from itertools import combinations

logger2 = logging.getLogger('_alt_logger')

class SensorManagerAlt(entities.entity.Entity):
    def __init__(self):
        super().__init__()
        self.microphones = []
        self.graph = nx.Graph()
        self.distance_threshold = 0.008
        self.speed_of_sound = 343  # Speed of sound in m/s

    def add_microphone(self, microphone):
        """Add a microphone to the sensor manager."""
        self.microphones.append(microphone)

    def build_graph(self):
        """Build a graph connecting microphones based on proximity."""
        for mic in self.microphones:
            self.graph.add_node(mic)

        for mic1, mic2 in combinations(self.microphones, 2):
            coord1, coord2 = mic1.getLLA()[:2], mic2.getLLA()[:2]
            if geodesic(coord1, coord2).km <= self.distance_threshold:
                self.graph.add_edge(mic1, mic2)

    def get_nearest_microphones(self, lat, lon, count):
        """Find the nearest microphones to a given location."""
        distances = [
            (mic, geodesic((lat, lon), mic.getLLA()[:2]).km)
            for mic in self.graph.nodes
        ]
        distances.sort(key=lambda x: x[1])
        return [mic for mic, _ in distances[:count]]

    def process_vocalization(self, animal):
        """Handle vocalization detection and triangulation."""
        lat, lon = animal.getLLA()[:2]
        nearby_mics = self.get_nearest_microphones(lat, lon, 4)

        for mic in nearby_mics:
            mic.set_trigger_event_time(
                animal.last_vocalisation_time + datetime.timedelta(seconds=mic.distance(animal) / self.speed_of_sound)
            )

        predicted_location, error = self.triangulate_location(nearby_mics, animal)

        for mic in nearby_mics:
            mic.reset_trigger_event_time()

        logger2.info(f'Triangulation error: {error} meters')
        return predicted_location, nearby_mics[0], error

    def triangulate_location(self, microphones, reference_animal):
        """Estimate the location of the animal using time difference of arrival."""
        def error_function(x):
            est_animal = Animal(species=Species("temp"))
            est_animal.setLLA(x)

            observed_deltas = [
                (mic.triggered_sim_clock_time - microphones[0].triggered_sim_clock_time).total_seconds()
                for mic in microphones
            ]
            estimated_deltas = [
                mic.distance(est_animal) / self.speed_of_sound - \
                microphones[0].distance(est_animal) / self.speed_of_sound
                for mic in microphones
            ]
            return np.sum((np.array(observed_deltas) - np.array(estimated_deltas))**2)

        initial_guess = np.mean([mic.getLLA()[:2] for mic in microphones], axis=0).tolist() + [10.0]
        bounds = (
            [self.get_region_bounds()["lat_min"], self.get_region_bounds()["lon_min"], 0],
            [self.get_region_bounds()["lat_max"], self.get_region_bounds()["lon_max"], np.inf]
        )

        result = minimize(
            error_function,
            initial_guess,
            method="L-BFGS-B",
            bounds=bounds
        )

        predicted_lat, predicted_lon, predicted_alt = result.x
        error = geodesic(reference_animal.getLLA()[:2], (predicted_lat, predicted_lon)).m

        return (predicted_lat, predicted_lon, predicted_alt), error

    def get_region_bounds(self):
        """Define the geographic bounds for triangulation."""
        return {
            "lat_min": -90,
            "lat_max": 90,
            "lon_min": -180,
            "lon_max": 180
        }
