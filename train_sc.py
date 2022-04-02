from RL.ambiguous_env import AmbiguousEnv
from RL.q_learning_single_core import QAgentSingleCore
from circuits.ioana_random import *
import sys
from logging import getLogger, INFO
from concurrent_log_handler import ConcurrentRotatingFileHandler
import os

log = getLogger(__name__)
logfile = os.path.abspath("./logfile_QTable.log")
rotateHandler = ConcurrentRotatingFileHandler(logfile, "a")
log.addHandler(rotateHandler)
log.setLevel(INFO)


def run():
    ep = 2000
    filename = sys.argv[1]
    moment_range = sys.argv[2]

    # episodes = [3, 5, 7, 10, 20, 50, 100, 500, 750, 1000]
    # partition_size = [3, 5, 7, 10, 15, 20, 30, 60, 80, 100]
    # max_iter = [5, 15, 30, 38, 50, 60, 75, 80, 100, 150]

    # Q_Table_actions = []
    # Q_Table_states = []
    # eps = []
    # parts = []
    # iterations = []

    # for ep in episodes:
    #     for p in partition_size:
    #         for it in max_iter:
    # starting_circuit = cirq.read_json(json_text=json_string)

    # starting_circuit = random.get_dummy_circuit(10, 1)

    f = open(f'train_circuits/{filename}', 'r')
    json_string = f.read()
    f.close()

    starting_circuit = cirq.read_json(json_text=json_string)

    print(starting_circuit)
    env = AmbiguousEnv(starting_circuit, filename, moment_range=int(moment_range))
    agent = QAgentSingleCore(env, n_ep=ep, max_iter=50, lr=0.01, gamma=0.97, expl_decay=0.001)
    agent.train()

    agent.show_evolution(filename='evolution.png', ep=ep)
                # filename = f'test.csv'

    #             Q_Table_states.append(agent.Q_table.shape[0])
    #             Q_Table_actions.append(agent.Q_table.shape[1])
    #             eps.append(ep)
    #             parts.append(p)
    #             iterations.append(it)
    #             log.info(f'{ep},{p},{it},{agent.Q_table.shape[0]},{agent.Q_table.shape[1]}')
    #
    # circopt_utils.plot_qt_size(eps, parts, iterations, Q_Table_states, 's')
    # circopt_utils.plot_qt_size(eps, parts, iterations, Q_Table_actions, 'a')


if __name__ == '__main__':
    run()
