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

    def __init__(self, circuit: cirq.Circuit, device_graph: nx.Graph, no_decomp_sets: int, nr_bits: int):
        self.circuit = circuit
        self.device_graph = device_graph
        self.no_decomp_sets = no_decomp_sets
        self.no_toffolis = self.get_number_of_toffolis()
        self.nr_bits = nr_bits
        self.configurations = []

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

    def get_random_decomposition_configuration(self):
        """
        Creates a list (size = no_decomp_sets) of unique Toffoli gates strategies to be applied at the decomposition step.
        """
        while len(self.configurations) < self.no_decomp_sets:
            sample: List[ToffoliDecompType] = self.get_random_sample()
            if sample not in self.configurations:
                self.configurations.append(sample)
        print('Generated configurations: ', self.configurations)

    def get_random_sample(self) -> List[ToffoliDecompType]:
        """
        :return: return random sample of strategies for the given circuit.
        """
        decompositions: List[ToffoliDecompType] = [ToffoliDecompType.RANDOM for toffoli in range(0, self.no_toffolis)]
        return decompositions

    def decompose_toffolis_in_circuit(self, strategies: List[ToffoliDecompType]) -> cirq.Circuit:
        """
        Toffoli gate decomposition of the given circuit by a list of strategies to be applied on each gate.
        :param strategies: List of strategies to be applied
        :return: decomposed circuit
        """
        new_circuit: cirq.Circuit = cirq.Circuit()
        strategy_count: int = 0
        for moment in self.circuit:
            had_toffoli: bool = False
            for op in moment.operations:
                if op.gate == cirq.TOFFOLI:
                    had_toffoli = True
                    qubits = op.qubits
                    moments = td.ToffoliDecomposition(
                        decomposition_type=strategies[strategy_count],
                        qubits=qubits).decomposition()
                    new_circuit.append(moments)
                    strategy_count += 1
                    break
            if not had_toffoli:
                new_circuit.append(moment)

        return new_circuit

    def route_circuit_for_multiple_configurations(self) -> List[List[Union[str, int]]]:
        """
        Routes given circuit for each random configuration generated.
        :return: List with results for each circuit - decomposition strategy list pair containing for each input:
         * number of adder bits
         * Toffoli gates decomposing configuration
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
            conf = "".join([str(strategy.value) for strategy in conf])
            csv_lines.append([self.nr_bits, conf, len(circuit.all_qubits()),
                              circuit_depth_before, len(circuit), process_time])

        return csv_lines
