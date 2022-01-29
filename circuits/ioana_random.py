import cirq
import random

def add_random_CNOT(circuit: cirq.Circuit, qubits):
    control = qubits[random.randint(0, len(qubits) - 1)]
    target = qubits[random.randint(0, len(qubits) - 1)]

    while control == target:
        target = qubits[random.randint(0, len(qubits) - 1)]

    circuit.append([cirq.CNOT.on(control, target)], strategy=cirq.InsertStrategy.NEW)
    return circuit


def add_random_H(circuit: cirq.Circuit, qubits):
    circuit.append([cirq.H.on(qubits[random.randint(0, len(qubits) - 1)])], strategy=cirq.InsertStrategy.NEW)
    return circuit


def get_random_circuit(nr_qubits: int, big_o_const: int):
    # TODO: Use maybe https://quantumai.google/reference/python/cirq/testing/random_circuit

    qubits = [cirq.NamedQubit(str(i)) for i in range(nr_qubits)]
    circuit = cirq.Circuit()

    degree = 4
    total_depth = big_o_const * nr_qubits ** degree
    for i in range(total_depth):
        if random.randint(1, 10) <= 4:
            circuit = add_random_CNOT(circuit, qubits)
        else:
            circuit = add_random_H(circuit, qubits)

    return circuit