import gym
import cirq
from RL.circuit_env_decomp import CircuitEnv
from quantify.mathematics.carry_ripple_4t_adder import CarryRipple4TAdder
import cirq.contrib.routing as ccr
import networkx as nx
from RL.q_learning import QAgent
from quantify.utils.counting_utils import count_toffoli_of_circuit
import numpy as np


def run():
    bits: int = 5
    device_graph: nx.Graph = ccr.get_grid_device_graph(20, 20)
    circuit: cirq.Circuit = CarryRipple4TAdder(bits).circuit
    env: gym.Env = CircuitEnv(circuit, device_graph)
    agent = QAgent(env, max_iter=count_toffoli_of_circuit(circuit))
    agent.train()
    agent.show_evolution()


if __name__ == '__main__':
    run()
