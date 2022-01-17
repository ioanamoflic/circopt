from typing import List, Tuple

import random

import cirq
from cirq import InsertStrategy

from RL.circuit_env_identities import CircuitEnvIdent
from RL.q_learning import QAgent
import routing.routing_multiple as rm
from circuits.bernstein import bernstein_vazirani
from circopt_utils import get_all_possible_identities
import sys

import global_stuff as g


def add_random_CNOT(circuit: cirq.Circuit, qubits):
    control = qubits[random.randint(0, len(qubits) - 1)]
    target = qubits[random.randint(0, len(qubits) - 1)]

    while control == target:
        target = qubits[random.randint(0, len(qubits) - 1)]

    circuit.append([cirq.CNOT.on(control, target)], strategy=InsertStrategy.NEW)
    return circuit


def add_random_H(circuit: cirq.Circuit, qubits):
    circuit.append([cirq.H.on(qubits[random.randint(0, len(qubits) - 1)])], strategy=InsertStrategy.NEW)
    return circuit


def get_random_circuit(nr_qubits: int, added_depth: int):
    # TODO: Use maybe https://quantumai.google/reference/python/cirq/testing/random_circuit

    qubits = [cirq.NamedQubit(str(i)) for i in range(nr_qubits)]
    circuit = cirq.Circuit()

    for i in range(added_depth):
        if random.randint(1, 10) <= 4:
            circuit = add_random_CNOT(circuit, qubits)
        else:
            circuit = add_random_H(circuit, qubits)

    return circuit


def run():
    ep = 3000
    # starting_circuit: cirq.Circuit = bernstein_vazirani(nr_bits=3, secret="110")
    # qbits = 3
    qubit_trials = [5, 16, 20, 25]
    depth_trials = [100, 150, 200, 250]
    run_identifier = sys.argv[1]

    random.seed(0)

    nr_qlearn_trials: int = 1
    for start in range(len(depth_trials)):
        qbits = qubit_trials[start:start+1][0]
        added_depth = depth_trials[start:start+1][0]
        starting_circuit = get_random_circuit(qbits, added_depth)
        print(starting_circuit)

        circ_dec = rm.RoutingMultiple(starting_circuit, no_decomp_sets=10, nr_bits=qbits)
        circ_dec.get_random_decomposition_configuration()

        for i in range(nr_qlearn_trials):
            conf: str = circ_dec.configurations.pop()
            decomposed_circuit = circ_dec.decompose_toffolis_in_circuit(conf)
            possible_identities, _ = get_all_possible_identities(decomposed_circuit)

            g.state_map_identity = dict()
            g.state_counter = dict()
            g.action_map = dict()

            env = CircuitEnvIdent(decomposed_circuit, could_apply_on=possible_identities)

            agent = QAgent(env, n_ep=ep, max_iter=2000, lr=0.01, gamma=0.97)
            agent.train(run_identifier, qbits)
            filename = str(run_identifier) + '_' + str(qbits) + '_qb_random.csv'
            agent.show_evolution(filename=filename, bvz_bits=qbits, ep=ep)


if __name__ == '__main__':
    run()
