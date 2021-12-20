from typing import List, Tuple

import gym
from gym import spaces
import numpy as np
import networkx as nx
import cirq
from gym.spaces import Discrete
import global_stuff as c
from quantify.qramcircuits.toffoli_decomposition import ToffoliDecompType
import quantify.qramcircuits.toffoli_decomposition as td
import circopt_utils
import quantify.optimizers as cnc
import quantify.utils.counting_utils as cu


class CircuitEnvDecomp(gym.Env):
    """
    A circuit environment for OpenAI Gym.
    """
    # is this useless?
    metadata = {'render.modes': ['human']}
    NO_ACTIONS = 5

    def __init__(self, starting_circuit: cirq.Circuit, device_graph: nx.Graph):
        super(CircuitEnvDecomp, self).__init__()
        self.current_action: int = 0
        self.current_config: str = ""

        self.no_toffs: int = cu.count_toffoli_of_circuit(starting_circuit)
        self.done: bool = False
        self.reward_range: Tuple[int, int] = (0, 1)
        self.starting_circuit: cirq.Circuit = starting_circuit
        self.current_circuit: cirq.Circuit = starting_circuit
        self.action_space: Discrete = spaces.Discrete(self.NO_ACTIONS)
        self.observation_space: Discrete = spaces.Discrete((pow(self.NO_ACTIONS, self.no_toffs + 1) - 1)
                                                           // (self.NO_ACTIONS - 1))
        self.device: nx.Graph = device_graph

    def _decompose_toffoli_in_circuit(self) -> bool:
        """
        de scris ceva destept aici.
        :return: decomposed circuit
        """
        new_circuit: cirq.Circuit = cirq.Circuit()
        toffoli_found: bool = False

        for moment in self.current_circuit:
            if circopt_utils.moment_has_toffoli(moment) and not toffoli_found:
                toffoli_found = True
                moments = td.ToffoliDecomposition.construct_decomposed_moments(cirq.Circuit(moment),
                                                                               ToffoliDecompType(
                                                                                   self.current_action))
                print(ToffoliDecompType(self.current_action))
                new_circuit.append(moments)
            else:
                new_circuit.append(moment)

        self.current_circuit = new_circuit
        return not toffoli_found

    def _optimize(self):
        cncl = cnc.CancelNghHadamards()
        cncl.optimize_circuit(self.current_circuit)

        drop_empty = cirq.optimizers.DropEmptyMoments()
        drop_empty.optimize_circuit(self.current_circuit)

        cncl = cnc.CancelNghCNOTs()
        cncl.optimize_circuit(self.current_circuit)

        drop_empty = cirq.optimizers.DropEmptyMoments()
        drop_empty.optimize_circuit(self.current_circuit)

    def step(self, action: int):
        self.current_action = action
        self.current_config = self.current_config + chr(ord('a') + action)

        c.state_map[self.current_config] = len(c.state_map)

        # 1. Update the environment state based on the chosen action
        depth_before = len(self.current_circuit)
        self.done = self._decompose_toffoli_in_circuit()
        self._optimize()
        depth_after = len(self.current_circuit)

        # 2. Calculate the "reward" for the new state of the circuit
        reward = depth_before / depth_after

        # 3. Store the new "observation" for the state (Toffoli config)
        observation = c.state_map.get(self.current_config)

        return observation, reward, self.done, {}

    def reset(self):
        self.current_circuit = self.starting_circuit
        self.current_action = None
        self.current_config = ""
        self.done = False

        return c.state_map[self.current_config]

    def render(self, mode='human', close=False):
        if self.current_action == 0:
            print('Setting up...')
            return
        print(f'Chosen action: {ToffoliDecompType(self.current_action)}')
        print(f'Current circuit depth: {len(self.current_circuit)}')
        print(f'Depth ratio: {len(self.starting_circuit) / len(self.current_circuit)}')
