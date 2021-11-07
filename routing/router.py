from typing import Tuple
import cirq
from cirq.contrib.routing import SwapNetwork
from cirq.contrib.routing.greedy import route_circuit_greedily
import networkx as nx
import time


class GreedyRouter:
    """
        Helper class for routing Quantify circuits.
    """
    def __init__(self, circuit: cirq.Circuit, device_graph: nx.Graph):
        self.circuit = circuit
        self.device_graph = device_graph

    def route(self) -> Tuple[cirq.Circuit, float]:
        """
        Routes circuit.
        :return: resulted circuit and process time in seconds.
        """
        start = time.process_time()
        result_circuit: SwapNetwork = route_circuit_greedily(self.circuit, self.device_graph)
        end = time.process_time()

        return result_circuit.circuit, end - start

