import random

from RL.circuit_env_identities import CircuitEnvIdent
from RL.q_learning import QAgent
import routing.routing_multiple as rm
from circuits.bernstein import bernstein_vazirani
from circuits.ioana_random import *
from circopt_utils import get_all_possible_identities
import sys

import global_stuff as g


def run():
    ep = 3000
    run_identifier = sys.argv[1]
    qubits = sys.argv[2]
    depth = sys.argv[3]

    random.seed(0)

    qubit_trials = [int(qubits)]
    depth_trials = [int(depth)]

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

            env = CircuitEnvIdent(decomposed_circuit)

            agent = QAgent(env, n_ep=ep, max_iter=1000, lr=0.01, gamma=0.97)
            agent.train(run_identifier, qbits)
            filename = f'{run_identifier}_{qbits}_qb_random.csv'
            agent.show_evolution(filename=filename, bvz_bits=qbits, ep=ep)


if __name__ == '__main__':
    run()
