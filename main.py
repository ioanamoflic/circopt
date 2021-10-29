import random
import cirq
import cirq.contrib.routing as ccr
from cirq.contrib.routing.greedy import route_circuit_greedily
# import networkx as nx
import matplotlib.pyplot as plt
import time
import numpy as np


def get_start_circuit() -> cirq.Circuit:
    start_circuit = cirq.Circuit()
    a, b = cirq.LineQubit.range(2)
    start_circuit.append(cirq.H(a))
    start_circuit.append(cirq.CNOT(a, b))

    return start_circuit


def add_qubit_to_circuit(circuit) -> cirq.Circuit:
    b = cirq.LineQubit(len(circuit.all_qubits()))
    circuit.append(cirq.H(b))

    return circuit


def add_random_cnot(circuit) -> cirq.Circuit:
    qub1 = random.randint(0, len(circuit.all_qubits()) - 1)
    qub2 = random.randint(0, len(circuit.all_qubits()) - 1)
    while qub1 == qub2:
        qub2 = random.randint(0, len(circuit.all_qubits()) - 1)

    a = cirq.LineQubit(qub1)
    b = cirq.LineQubit(qub2)
    circuit.append(cirq.CNOT(a, b))

    return circuit


def get_moments(circuit):
    for moment in circuit:
        print(moment)


def depth(circuit) -> int:
    return len(cirq.Circuit(circuit.all_operations()))


def main():
    # device_graph = ccr.get_grid_device_graph(20, 20)
    # nx.draw(device_graph)
    # plt.show()

    no_qubits = []
    depths = np.zeros(3)
    times = np.zeros(3)
    gate_domain = {cirq.ops.CNOT: 2}

    for i in range(0, 3):
        for j in range(2, 5):

            device_graph = ccr.get_linear_device_graph(j)
            circuit = cirq.testing.random_circuit(j, 15, 0.5, gate_domain, random_state=37)
            depth_before = depth(circuit)

            start = time.process_time()
            swaps = route_circuit_greedily(circuit, device_graph)
            times[j - 2] += (time.process_time() - start)

            depth_after = depth(swaps.circuit)
            depths[j - 2] += (depth_before / depth_after)

            if i == 0:
                no_qubits.append(j)

    fig, axs = plt.subplots(2)
    fig.suptitle('max_search_radius = 1, dummy examples')
    axs[0].set_xlabel('No. qubits')
    axs[0].set_ylabel('Av. depth ratio')
    axs[0].plot(no_qubits, depths / 3)

    axs[1].set_xlabel('No. qubits')
    axs[1].set_ylabel('Av. time (seconds)')
    axs[1].plot(no_qubits, times / 3)

    plt.show()


if __name__ == '__main__':
    main()
