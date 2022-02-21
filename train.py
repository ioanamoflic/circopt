import circopt_utils
from RL.circuit_env_identities import CircuitEnvIdent
from RL.multi_env import SubprocVecEnv
from RL.q_learning import QAgent
from circuits.ioana_random import *
import sys


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


def make_mp_envs(env_id, num_env, seed, json_string, start_idx=0):
    def make_env(rank):
        def fn():
            starting_circuit = cirq.read_json(json_text=json_string)
            env = CircuitEnvIdent(starting_circuit)
            env.seed(seed + rank)
            return env

        return fn

    return SubprocVecEnv([make_env(i + start_idx) for i in range(num_env)])


def run():
    ep = 3000
    circuit_file = sys.argv[1]

    random.seed(0)

    f = open(circuit_file, 'r')
    json_string = f.read()
    vec_env = make_mp_envs(None, 2, random.randint(0, 7), json_string)
    agent = QAgent(vec_env, n_ep=ep, max_iter=50, lr=0.01, gamma=0.97)
    agent.train()
    filename = f'test.csv'
    agent.show_evolution(filename=filename, bvz_bits=15, ep=ep)

    circopt_utils.plot_reward_function()


if __name__ == '__main__':
    run()
