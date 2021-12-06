from typing import Tuple, List

import networkx as nx
import numpy as np

import gym
from gym import spaces
from gym.spaces import Discrete

import cirq

import config as c

from optimization.TopLeftT import TopLeftT
from optimization.TopRightT import TopRightT
from optimization.TopLeftHadamard import TopLeftHadamard
from optimization.OneHLeft2Right import OneHLeftTwoRight

import circopt_utils
import quantify.optimizers as cnc


class CircuitEnvIdent(gym.Env):
    """
    An OpenAI Gym circuit environment.
    """
    # is this useless?
    metadata = {'render.modes': ['human']}
    NO_ACTIONS = 2

    def __init__(self, starting_circuit: cirq.Circuit, could_apply_on, device_graph: nx.Graph = None):
        super(CircuitEnvIdent, self).__init__()
        self.current_action: int = 0
        self.starting_circuit: cirq.Circuit = starting_circuit
        self.current_circuit: cirq.Circuit = starting_circuit
        self.done: bool = False
        self.current_config: List[int] = [0 for i in range(len(self.current_circuit))]
        c.state_map_identity[circopt_utils.to_str(self.current_config)] = len(c.state_map_identity)
        self.current_moment = 0
        self.reward_range: Tuple[int, int] = (0, 1)
        self.action_space: Discrete = spaces.Discrete(2)
        self.observation_space: Discrete = spaces.Discrete((pow(self.NO_ACTIONS, len(self.starting_circuit))))
        self.device: nx.Graph = device_graph
        self.could_apply_on: List[Tuple[int, int]] = could_apply_on
        self.initial_degree: float = self._get_circuit_degree()
        # self.first_entry = True
        # self.history_rewards = []
        # self.reward_decay = 0.1

    def _apply_identity(self, action):
        if action == 0:
            return

        if action == 1:
            self.current_config[self.could_apply_on[self.current_moment][1]] = self.could_apply_on[self.current_moment][0]

            if self.could_apply_on[self.current_moment][0] == 1:
                opt_circuit = TopLeftHadamard(where_to=self.could_apply_on[self.current_moment][1])
                opt_circuit.optimize_circuit(self.current_circuit)
                self._optimize()
                return

            if self.could_apply_on[self.current_moment][0] == 2:
                opt_circuit = OneHLeftTwoRight(where_to=self.could_apply_on[self.current_moment][1])
                opt_circuit.optimize_circuit(self.current_circuit)
                self._optimize()
                return

            if self.could_apply_on[self.current_moment][0] == 3:
                opt_circuit = TopRightT(where_to=self.could_apply_on[self.current_moment][1])
                opt_circuit.optimize_circuit(self.current_circuit)
                self._optimize()
                return

            if self.could_apply_on[self.current_moment][0] == 4:
                opt_circuit = TopLeftT(where_to=self.could_apply_on[self.current_moment][1])
                opt_circuit.optimize_circuit(self.current_circuit)
                self._optimize()

    def _optimize(self):
        cncl = cnc.CancelNghHadamards()
        cncl.optimize_circuit(self.current_circuit)

        cncl = cnc.CancelNghCNOTs()
        cncl.optimize_circuit(self.current_circuit)

        drop_empty = cirq.optimizers.DropEmptyMoments()
        drop_empty.optimize_circuit(self.current_circuit)

    def _get_circuit_degree(self):
        degrees = np.zeros(len(self.current_circuit.all_qubits()))
        qubit_dict = {}
        for qubit in self.current_circuit.all_qubits():
            qubit_dict[qubit] = len(qubit_dict)

        for moment in self.current_circuit:
            for op in moment:
                if isinstance(op, cirq.GateOperation) and op.gate == cirq.CNOT:
                    q1 = qubit_dict[op.qubits[0]]
                    q2 = qubit_dict[op.qubits[1]]
                    degrees[q1: q1 + 1] += 1
                    degrees[q2: q2 + 1] += 1

        return np.mean(degrees)

    def step(self, action: int):
        # update possible identity for each moment at current time step (for identities that add moments)
        # max_iter_episode should change
        # get_all_possible_identities for each step (?)
        # self.could_apply_on = utils.get_all_possible_identities(self.current_circuit)
        # self.current_moment = ...

        self.current_action = action

        # 1. Update the environment state based on the chosen action
        self._apply_identity(action)

        # 2. Calculate the "reward" for the new state of the circuit
        reward = self.initial_degree / self._get_circuit_degree()
        print(reward)

        # 3. Store the new "observation" for the state (Identity config)
        # config_as_str: str = circopt_utils.to_str(self.current_config)
        #
        # Alexandru: e unic?
        #
        n_circuit = cirq.Circuit(self.current_circuit.all_operations(), strategy=cirq.InsertStrategy.EARLIEST)
        config_as_str = str(n_circuit)

        # keep initial position of state in QTable
        if config_as_str not in c.state_map_identity.keys():
            c.state_map_identity[config_as_str] = len(c.state_map_identity)

        observation: int = c.state_map_identity.get(config_as_str)

        self.current_moment += 1

        return observation, reward, self.done, {}

    def reset(self):
        self.current_circuit = self.starting_circuit
        self.current_action = None
        self.current_moment = 0
        self.current_config = [0 for i in range(len(self.starting_circuit))]
        config_as_str = circopt_utils.to_str(self.current_config)

        if config_as_str not in c.state_map_identity.keys():
            c.state_map_identity[config_as_str] = len(c.state_map_identity)
        self.done = False

        return c.state_map_identity[config_as_str]

    def render(self, mode='human', close=False):
        if self.current_action == 0:
            print('Setting up...')
            return
        print(f'Chosen action: {self.current_action}')
        print(f'Current circuit depth: {len(self.current_circuit)}')
        print(f'Depth ratio: {len(self.starting_circuit) / len(self.current_circuit)}')
