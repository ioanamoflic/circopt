from typing import Tuple, List

import networkx as nx
import numpy as np

import gym
from gym import spaces
from gym.spaces import Discrete

import cirq
import global_stuff as g
from optimization.StickMultiTargetToCNOT import StickMultiTargetToCNOT

from optimization.TopLeftHadamard import TopLeftHadamard
from optimization.OneHLeft2Right import OneHLeftTwoRight
from optimization.ReverseCNOT import ReverseCNOT
from optimization.HadamardSquare import HadamardSquare
from optimization.StickCNOTs import StickCNOTs
from optimization.StickMultiTarget import StickMultiTarget

import circopt_utils
import quantify.optimizers as cnc
from optimization.optimize_circuits import CircuitIdentity
import copy


class CircuitEnvIdent(gym.Env):
    """
    An OpenAI Gym circuit environment.
    """
    # is this useless?
    metadata = {'render.modes': ['human']}
    NO_ACTIONS = 2

    def __init__(self, starting_circuit: cirq.Circuit, could_apply_on, device_graph: nx.Graph = None):
        super(CircuitEnvIdent, self).__init__()
        self.current_action: Tuple[int, int, int] = (0, 0, 0)
        self.starting_circuit: cirq.Circuit = starting_circuit
        self.current_circuit: cirq.Circuit = copy.deepcopy(self.starting_circuit)
        self.done: bool = False
        self.len_start = len(self.starting_circuit)
        g.state_map_identity[circopt_utils.get_unique_representation(self.current_circuit)] = len(g.state_map_identity)
        g.current_moment = 0
        self.reward_range: Tuple[float, float] = (0.0, 3.0)
        self.action_space: Discrete = spaces.Discrete(2)
        self.observation_space: Discrete = spaces.Discrete((pow(self.NO_ACTIONS, len(self.starting_circuit))))
        self.could_apply_on: List[Tuple[int, int]] = could_apply_on
        self.initial_degree: float = self._get_circuit_degree()
        self.episode_is_done = False

    def _apply_identity(self, action):
        if action == 0:
            g.current_moment = self.could_apply_on[0][1] + 1
            return

        if action == 1:
            if self.could_apply_on[0][0] == CircuitIdentity.ONE_HADAMARD_UP_LEFT:
                opt_circuit = TopLeftHadamard(where_to=self.could_apply_on[0][1])
                opt_circuit.optimize_circuit(self.current_circuit)
                g.current_moment = self.could_apply_on[0][1] + 2
                return

            if self.could_apply_on[0][0] == CircuitIdentity.ONE_HADAMARD_LEFT_DOUBLE_RIGHT:
                opt_circuit = OneHLeftTwoRight(where_to=self.could_apply_on[0][1])
                opt_circuit.optimize_circuit(self.current_circuit)
                g.current_moment = self.could_apply_on[0][1] - 1
                return

            if self.could_apply_on[0][0] == CircuitIdentity.DOUBLE_HADAMARD_LEFT_RIGHT:
                opt_circuit = HadamardSquare(where_to=self.could_apply_on[0][1])
                opt_circuit.optimize_circuit(self.current_circuit)
                g.current_moment = self.could_apply_on[0][1] - 2
                return

            if self.could_apply_on[0][0] == CircuitIdentity.REVERSED_CNOT:
                opt_circuit = ReverseCNOT(where_to=self.could_apply_on[0][1])
                opt_circuit.optimize_circuit(self.current_circuit)
                g.current_moment = self.could_apply_on[0][1] + 2

    def _optimize(self):
        len_circ_before = len(self.current_circuit)
        cncl = cnc.CancelNghHadamards(optimize_till=g.current_moment)
        cncl.optimize_circuit(self.current_circuit)

        cncl = cnc.CancelNghCNOTs(optimize_till=g.current_moment)
        cncl.optimize_circuit(self.current_circuit)

        cncl = StickCNOTs(optimize_till=g.current_moment + 1)
        cncl.optimize_circuit(self.current_circuit)

        cncl = StickMultiTarget(optimize_till=g.current_moment + 1)
        cncl.optimize_circuit(self.current_circuit)

        opt_circuit = StickMultiTargetToCNOT(optimize_till=g.current_moment + 1)
        opt_circuit.optimize_circuit(self.current_circuit)

        drop_empty = cirq.optimizers.DropEmptyMoments()
        drop_empty.optimize_circuit(self.current_circuit)

        len_circ_after = len(self.current_circuit)

        g.current_moment -= (len_circ_before - len_circ_after)

        cncl = cnc.CancelNghHadamards()
        cncl.optimize_circuit(self.current_circuit)

        cncl = cnc.CancelNghCNOTs()
        cncl.optimize_circuit(self.current_circuit)

        cncl = StickMultiTarget()
        cncl.optimize_circuit(self.current_circuit)

        opt_circuit = StickMultiTargetToCNOT()
        opt_circuit.optimize_circuit(self.current_circuit)

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

    def _len_move_to_left(self):
        n_circuit = cirq.Circuit(self.current_circuit.all_operations(), strategy=cirq.InsertStrategy.EARLIEST)
        return len(n_circuit)

    def start_again(self):
        g.current_moment = 0
        self.could_apply_on = circopt_utils.get_all_possible_identities(self.current_circuit)
        self.could_apply_on = [i for i in self.could_apply_on if i[1] >= g.current_moment]

    def step(self, action: int):

        print('Possible identities: ', self.could_apply_on)

        # 1. Update the environment state based on the chosen action
        self.current_action = circopt_utils.get_action_by_value(action)
        print('current action: ', self.current_action)

        print('current moment before: ', g.current_moment)
        self._apply_identity(self.current_action[2])
        print('circuit after identity: \n', self.current_circuit)

        self._optimize()
        print('circuit after optimization: \n', self.current_circuit)
        print('current moment after: ', g.current_moment)

        # 2. Calculate the "reward" for the new state of the circuit
        # reward = self.initial_degree / self._get_circuit_degree()
        reward = self.len_start / self._len_move_to_left()
        print('reward: ', reward)
        # make agent avoid starting state
        if circopt_utils.get_unique_representation(self.starting_circuit) == \
            circopt_utils.get_unique_representation(self.current_circuit):
            reward = 0.0

        print(reward)

        # 3. Store the new "observation" for the state (Identity config)
        circuit_as_string = circopt_utils.get_unique_representation(self.current_circuit)

        # keep initial position of state in QTable
        if circuit_as_string not in g.state_map_identity.keys():
            g.state_map_identity[circuit_as_string] = len(g.state_map_identity)

        # count every time a state is encountered
        g.state_counter[circuit_as_string] = g.state_counter.get(circuit_as_string, 0) + 1

        observation: int = g.state_map_identity.get(circuit_as_string)

        # will be replaced with a future 'get_next_identity'
        self.could_apply_on = circopt_utils.get_all_possible_identities(self.current_circuit)
        self.could_apply_on = [i for i in self.could_apply_on if i[1] >= g.current_moment]

        if not self.could_apply_on:
            self.done = True

        return observation, reward, self.done, {}

    def reset(self):
        self.current_circuit = copy.deepcopy(self.starting_circuit)
        g.current_moment = 0
        self.done = False
        self.could_apply_on = circopt_utils.get_all_possible_identities(self.current_circuit)
        self.could_apply_on = [i for i in self.could_apply_on if i[1] >= g.current_moment]

        circuit_as_string = circopt_utils.get_unique_representation(self.current_circuit)
        if circuit_as_string not in g.state_map_identity.keys():
            g.state_map_identity[circuit_as_string] = len(g.state_map_identity)

        return g.state_map_identity.get(circuit_as_string)

    def render(self, mode='human', close=False):
        if self.current_action == 0:
            print('Setting up...')
            return
        print(f'Chosen action: {self.current_action}')
        print(f'Current circuit depth: {len(self.current_circuit)}')
        print(f'Depth ratio: {len(self.starting_circuit) / len(self.current_circuit)}')
