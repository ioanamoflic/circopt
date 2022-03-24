from RL.circuit_env_identities import CircuitEnvIdent
from RL.multi_env import SubprocVecEnv
from RL.q_learning import QAgent
from circuits.ioana_random import *
import sys
import fnmatch
import os
import circopt_utils as utils


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


def make_mp_envs(num_env, seed, circuits, moment_range=10, start_idx=0):
    def make_env(rank, circuit):
        def fn():
            starting_circuit = cirq.read_json(json_text=circuit[0])
            env = CircuitEnvIdent(starting_circuit, circuit[1], moment_range)
            env.seed(seed + rank)
            env.ep = -1
            return env

        return fn

    return SubprocVecEnv([make_env(i + start_idx, circuits[i]) for i in range(num_env)])


def run():
    ep = 3000
    file_prefix = sys.argv[1]
    batch_number = sys.argv[2]
    random.seed(0)

    circuits = []
    for file in os.listdir('./train_circuits'):
        if fnmatch.fnmatch(file, f'{file_prefix.upper()}_{batch_number}*.txt'):
            f = open(f'train_circuits/{file}', 'r')
            json_string = f.read()
            circuits.append((json_string, file))
            f.close()

    vec_env = make_mp_envs(num_env=7, seed=random.randint(0, 15), circuits=circuits, moment_range=5)
    agent = QAgent(vec_env, n_ep=ep, max_iter=50, lr=0.01, gamma=0.97, expl_decay=0.001)
    agent.train()

    # utils.plot_qt_size(eps, parts, exps, Q_Table_states, 's')
    # utils.plot_qt_size(eps, parts, exps, Q_Table_actions, 'a')


if __name__ == '__main__':
    run()