from mic_generator import Map
import networkx as nx
import matplotlib.pyplot as plt
from event_generator import Event
from geopy.distance import distance
from math import radians, sin, cos, sqrt, atan2, degrees

class Simulator():
    def __init__(self) -> None:
        self.NODE_THRESHOLD = 0.005
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
        nx.draw(self.graph, with_labels=True)
        plt.show()

    def triangulate_event(self, event_lat: float, event_long: float) -> None:
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

        event_lat_est, event_long_est = self._triangulate(toda_data)
        error = distance((event_lat, event_long), (event_lat_est, event_long_est)).m
        print(f"Triangulation error: {error} meters")

    def _triangulate(self, tdoa_data: list[tuple[float, float, float]]) -> tuple[float, float]:
        # Speed of sound in air, in meters per second
        c = 343

        # Convert latitudes and longitudes to radians
        locations = [(radians(lat), radians(lon), t) for lat, lon, t in tdoa_data]

        # Compute the differences in TOA between each pair of microphones
        delta_t = [(t2 - t1) for (lat1, lon1, t1), (lat2, lon2, t2) in zip(locations, locations[1:])]

        # Convert latitudes and longitudes to Cartesian coordinates
        x = [(cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)) for lat, lon, t in locations]

        # Compute the direction vector between each pair of microphones
        v = [(x2[0]-x1[0], x2[1]-x1[1], x2[2]-x1[2]) for x1, x2 in zip(x, x[1:])]

        # Compute the dot product of each direction vector with itself
        vv = [(v1[0]**2 + v1[1]**2 + v1[2]**2, v2[0]**2 + v2[1]**2 + v2[2]**2) for v1, v2 in zip(v, v[1:])]

        # Compute the cross product of each pair of direction vectors
        cp = [(v1[1]*v2[2]-v1[2]*v2[1], v1[2]*v2[0]-v1[0]*v2[2], v1[0]*v2[1]-v1[1]*v2[0]) for v1, v2 in zip(v, v[1:])]

        # Compute the norm of the cross product
        norm_cp = [sqrt(c1**2 + c2**2 + c3**2) for c1, c2, c3 in cp]

        # Compute the distances between each pair of microphones
        distances = [c * dt for dt in delta_t]

        # Compute the possible locations of the event
        possible_locations = []
        for i in range(len(locations) - 2):
            for j in range(i + 1, len(locations) - 1):
                for k in range(j + 1, len(locations)):
                    # Compute the coordinates of the possible location
                    A = vv[i][0] + vv[j][0] - vv[k][0]
                    B = 2 * (cp[i][0] + cp[j][0] - cp[k][0])
                    C = vv[i][1] + vv[j][1] - vv[k][1]

                    D = 2 * (cp[i][1] + cp[j][1] - cp[k][1])
                    E = vv[i][2] + vv[j][2] - vv[k][2]
                    F = 2 * (cp[i][2] + cp[j][2] - cp[k][2])

                    G = distances[k]**2 - distances[j]**2 - locations[k][0]**2 - locations[k][1]**2 + locations[j][0]**2 + locations[j][1]**2 + distances[j]**2 - distances[i]**2 - locations[j][0]**2 - locations[j][1]**2 + locations[i][0]**2 + locations[i][1]**2

                    # Compute the coefficients of the quadratic equation
                    a = B**2 + D**2 + F**2
                    b = 2 * (A*B + C*D + E*F)
                    c = A**2 + C**2 + E**2 - G**2 - 2*(B*locations[i][0] + D*locations[i][1] + F*locations[i][2])
                    
                    # Compute the discriminant of the quadratic equation
                    discriminant = b**2 - 4*a*c

                    # If the discriminant is negative, there are no real solutions
                    if discriminant < 0:
                        continue

                    # Compute the two solutions of the quadratic equation
                    t1 = (-b + sqrt(discriminant)) / (2*a)
                    t2 = (-b - sqrt(discriminant)) / (2*a)

                    # Compute the possible locations of the event
                    possible_location_1 = (
                        locations[i][0] + t1*v[i][0],
                        locations[i][1] + t1*v[i][1]
                    )
                    possible_location_2 = (
                        locations[i][0] + t2*v[i][0],
                        locations[i][1] + t2*v[i][1]
                    )

                    # Add the possible locations to the list
                    possible_locations.append(possible_location_1)
                    possible_locations.append(possible_location_2)

        # Find the center of mass of the possible locations
        center_lat = sum([p[0] for p in possible_locations]) / len(possible_locations)
        center_lon = sum([p[1] for p in possible_locations]) / len(possible_locations)

        # Convert the center of mass back to degrees
        center_lat_deg = degrees(center_lat)
        center_lon_deg = degrees(center_lon)

        # Return the center of mass
        return (center_lat_deg, center_lon_deg)


if __name__ == "__main__":
    ECHO_SIMULATOR = Simulator()
    # ECHO_SIMULATOR.simulated_map.print_map()
    ECHO_SIMULATOR._connect_mics()
    # ECHO_SIMULATOR._draw_node_graph()
    ECHO_SIMULATOR.triangulate_event(*Event(ECHO_SIMULATOR.simulated_map).get_event_lat_long())