from RL.circuit_env_identities import CircuitEnvIdent
from RL.q_learning import QAgent
import routing.routing_multiple as rm
from circuits.bernstein import bernstein_vazirani
from circuits.ioana_random import *
from circopt_utils import get_all_possible_identities

import global_stuff as globals

def benchmark_single_parallelisation():

    print("compile...")
    # starting_circuit = get_random_circuit(nr_qubits=10, big_o_const=10)
    starting_circuit = get_test_circuit()

    for i in range(0, 2):
        print(f"optimize...parallel={i}")
        from optimization.reverse_CNOT import ReverseCNOT
        my_opt = ReverseCNOT()
        if i == 1:
            my_opt.go_parallel = True

        from time import time
        ts = time()
        my_opt.optimize_circuit(starting_circuit)
        te = time()

        print(f"{te - ts}seconds")

    return


def benchmark_parallelisation():

    print("compile...")

    for i in range(0, 2):
        globals.enablePrint()
        print(f"optimize...parallel={i}")

        if i == 0:
            globals.deparallelise_optimizers()
        elif i == 1:
            globals.parallelise_optimizers()

        globals.blockPrint()
        run(nr_episodes=2, max_iter = 10, run_identifier = 0, qubits = 5, constant = 1)

    return

@globals.timing
def run(nr_episodes = 2000, max_iter = 100, run_identifier = 0, qubits = 5, constant = 1):
    qubit_trials = [int(qubits)]
    constant_trials = [int(constant)]

    nr_qlearn_trials: int = 1
    for start in range(len(constant_trials)):
        qbits = qubit_trials[start:start+1][0]
        added_depth = constant_trials[start:start+1][0]
        # starting_circuit = get_random_circuit(qbits, added_depth)
        starting_circuit = get_test_circuit()
        print("Starting circuit: \n", starting_circuit)

        circ_dec = rm.RoutingMultiple(starting_circuit, no_decomp_sets=10, nr_bits=qbits)
        circ_dec.get_random_decomposition_configuration()

        for i in range(nr_qlearn_trials):
            conf: str = circ_dec.configurations.pop()
            decomposed_circuit = circ_dec.decompose_toffolis_in_circuit(conf)

            globals.state_map_identity = dict()
            globals.state_counter = dict()
            globals.action_map = dict()

            env = CircuitEnvIdent(decomposed_circuit)

            agent = QAgent(env, n_ep=nr_episodes, max_iter=max_iter, lr=0.01, gamma=0.97)
            agent.train(run_identifier, qbits)

            # filename = f'{run_identifier}_{qbits}_qb_random.csv'
            # agent.show_evolution(filename=filename, bvz_bits=qbits, ep=nr_episodes)


if __name__ == '__main__':
    random.seed(0)
    # run(run_identifier = sys.argv[1], qubits = sys.argv[2], constant = sys.argv[3])
    # benchmark_single_parallelisation()
    benchmark_parallelisation()