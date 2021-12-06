from typing import List, Union

import cirq
import networkx as nx
from quantify.qramcircuits.toffoli_decomposition import ToffoliDecompType
import quantify.qramcircuits.toffoli_decomposition as td
from routing.router import GreedyRouter


class RoutingMultiple:
    """
    Helper class for routing circuits on multiple Toffoli gate decompositions.
    """

    def __init__(self, circuit: cirq.Circuit, device_graph: nx.Graph = None, no_decomp_sets: int = 100, nr_bits: int = 4):
        self.circuit = circuit
        self.device_graph = device_graph
        self.no_decomp_sets = no_decomp_sets
        self.no_toffolis = self.get_number_of_toffolis()
        self.nr_bits = nr_bits
        self.configurations = set()

    def get_number_of_toffolis(self) -> int:
        """
        :return: number of Toffoli gates in a circuit
        """
        counter: int = 0
        for moment in self.circuit:
            for op in moment:
                if op.gate == cirq.TOFFOLI:
                    counter += 1

        return counter

    def get_random_decomposition_configuration(self, not_random = False) -> None:
        """
        Creates a set (size = no_decomp_sets) of unique strategies to be applied at the decomposition step.
        """
        while len(self.configurations) < self.no_decomp_sets:
            self.configurations.add(self.get_random_sample())

    def get_random_sample(self) -> str:
        """
        :return: random sample of strategies for the given circuit in the form of a string.
        """
        decompositions: str = "".join(
            [chr(ord('a') + ToffoliDecompType.RANDOM.value) for gate in range(self.no_toffolis)])
        return decompositions

    def decompose_toffolis_in_circuit(self, strategies: str) -> cirq.Circuit:
        """
        Toffoli gate decomposition of the given circuit by a list of strategies to be applied on each gate.
        :param strategies: strategies to be applied
        :return: decomposed circuit
        """
        new_circuit: cirq.Circuit = cirq.Circuit()
        strategy_count: int = 0

        for moment in self.circuit:

            has_toffoli: bool = False

            for op in moment.operations:
                if op.gate == cirq.TOFFOLI:
                    has_toffoli = True
                    break

            if has_toffoli:
                moments = td.ToffoliDecomposition.construct_decomposed_moments(cirq.Circuit(moment),
                            ToffoliDecompType(ord(strategies[strategy_count]) - ord('a') + 1))
                new_circuit.append(moments)

                #what if there is more than one toffoli / moment? different decompositions for each?
                strategy_count += 1
            else:
                new_circuit.append(moment)

        return new_circuit

    def route_circuit_for_multiple_configurations(self) -> List[List[Union[str, int]]]:
        """
        Routes given circuit for each random configuration generated.
        :return: List with results for each circuit - decomposition strategy pair containing for each input:
         * number of adder bits
         * Toffoli gates decomposition configuration
         * number of qubits of the output circuit
         * depth of the input circuit
         * depth of the output circuit
        """
        csv_lines: List[List[Union[str, int]]] = []
        circuit_depth_before: int = len(self.circuit)
        self.get_random_decomposition_configuration()

        for conf in self.configurations:
            circuit = self.decompose_toffolis_in_circuit(conf)
            circuit, process_time = GreedyRouter(circuit, self.device_graph).route()
            csv_lines.append([self.nr_bits, conf, len(circuit.all_qubits()),
                              circuit_depth_before, len(circuit), process_time])

        return csv_lines
