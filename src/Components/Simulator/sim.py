from mic_generator import Map
import networkx as nx
import matplotlib.pyplot as plt


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
                    distance = ((lat1 - lat2) ** 2 + (long1 - long2) ** 2) ** 0.5
                    if distance <= self.NODE_THRESHOLD:
                        self.graph.add_edge(mic1, mic2)

    def _draw_node_graph(self) -> None:
        nx.draw(self.graph, with_labels=True)
        plt.show()

if __name__ == "__main__":
    ECHO_SIMULATOR = Simulator()
    # ECHO_SIMULATOR.simulated_map.print_map()
    ECHO_SIMULATOR._connect_mics()
    ECHO_SIMULATOR._draw_node_graph()
    