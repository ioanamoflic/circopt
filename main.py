import cirq

from RL.circuit_env_identities import CircuitEnvIdent
from RL.q_learning import QAgent
import routing.routing_multiple as rm
from circuits.bernstein import bernstein_vazirani
from circuits.ioana_random import *
from circopt_utils import get_all_possible_identities

import sys

import global_stuff as g

def benchmark_parallelisation():

    print("compile...")
    starting_circuit = get_random_circuit(nr_qubits=10, big_o_const=10)

    from optimization.parallel_point_optimizer import ParallelPointOptimizer

    paralel = ParallelPointOptimizer()

    for i in range(2):
        print(f"optimize...parallel={i}")
        from optimization.reverse_CNOT import ReverseCNOT
        my_opt = ReverseCNOT()
        if i == 1:
            my_opt.optimize_circuit = paralel.optimize_circuit

        from time import time
        ts = time()
        my_opt.optimize_circuit(starting_circuit)
        te = time()

        print(f"{te - ts}seconds")

    return


def run():
    ep = 10
    circuit_file = sys.argv[1]
    # qubits = sys.argv[2]
    # constant = sys.argv[3]

    random.seed(0)

    # qubit_trials = [int(qubits)]
    # constant_trials = [int(constant)]

    nr_qlearn_trials: int = 1
    # for start in range(len(constant_trials)):
    #     qbits = qubit_trials[start:start+1][0]
    #     added_depth = constant_trials[start:start+1][0]
    #     # starting_circuit = get_random_circuit(qbits, added_depth)
    #     starting_circuit = get_test_circuit()
    f = open(circuit_file, 'r')
    json_string = f.read()
    starting_circuit = cirq.read_json(json_text=json_string)
    qbits = len(cirq.Circuit(starting_circuit).all_qubits())

    circuit_name = circuit_file.split('.txt')[0]

    print("Starting circuit: \n", starting_circuit)

    circ_dec = rm.RoutingMultiple(starting_circuit, no_decomp_sets=10)
    circ_dec.get_random_decomposition_configuration()

    for i in range(nr_qlearn_trials):
        conf: str = circ_dec.configurations.pop()
        decomposed_circuit = circ_dec.decompose_toffolis_in_circuit(conf)

        g.state_map_identity = dict()
        g.state_counter = dict()
        g.action_map = dict()

        env = CircuitEnvIdent(decomposed_circuit)

        agent = QAgent(env, n_ep=ep, max_iter=50, lr=0.01, gamma=0.97)
        agent.train(circuit_name)
        filename = f'{circuit_name}.csv'
        agent.show_evolution(filename=filename, bvz_bits=qbits, ep=ep)


if __name__ == '__main__':
    run()
