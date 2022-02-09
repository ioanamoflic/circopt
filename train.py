import cirq

import circopt_utils
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
    ep = 3000
    circuit_file = sys.argv[1]
    # qubits = sys.argv[2]
    # constant = sys.argv[3]

    random.seed(0)

    f = open(circuit_file, 'r')
    json_string = f.read()
    starting_circuit = cirq.read_json(json_text=json_string)
    qbits = len(cirq.Circuit(starting_circuit).all_qubits())

    circuit_file = circuit_file.split('.txt')[0]
    circuit_name = circuit_file.split('train_circuits/')[1]

    print("Starting circuit: \n", starting_circuit)

    g.state_map_identity = dict()
    g.state_counter = dict()
    g.action_map = dict()

    env = CircuitEnvIdent(starting_circuit)

    agent = QAgent(env, n_ep=ep, max_iter=200, lr=0.01, gamma=0.97)
    agent.train(circuit_name)
    filename = f'{circuit_file}.csv'
    agent.show_evolution(filename=filename, bvz_bits=qbits, ep=ep)

    circopt_utils.plot_reward_function()


if __name__ == '__main__':
    run()
