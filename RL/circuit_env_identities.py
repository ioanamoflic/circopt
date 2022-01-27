from typing import Tuple, List

import networkx as nx
import numpy as np

import gym
from gym import spaces
from gym.spaces import Discrete

import cirq
import global_stuff as g
from optimization.stick_multitarget_to_CNOT import StickMultiTargetToCNOT

from optimization.top_left_hadamard import TopLeftHadamard
from optimization.one_H_left_2_right import OneHLeftTwoRight
from optimization.reverse_CNOT import ReverseCNOT
from optimization.hadamard_square import HadamardSquare
from optimization.stick_CNOTs import StickCNOTs
from optimization.stick_multitarget import StickMultiTarget

import circopt_utils
import quantify.optimizers as cnc
import quantify.utils.misc_utils as mu
from optimization.optimize_circuits import CircuitIdentity
import logging
import copy
import time

logging.basicConfig(filename='logfile.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s')


class CircuitEnvIdent(gym.Env):
    """
    An OpenAI Gym circuit environment.
    """
    metadata = {'render.modes': ['human']}
    NO_ACTIONS = 2

    def __init__(self, starting_circuit: cirq.Circuit):
        super(CircuitEnvIdent, self).__init__()
        self.current_action: Tuple[int, int, int] = (0, 0, 0)
        self.starting_circuit: cirq.Circuit = starting_circuit
        self.current_circuit: cirq.Circuit = copy.deepcopy(self.starting_circuit)
        self.done: bool = False
        self.len_start: int = self._len_move_to_left()
        self.could_apply_on: List[Tuple[int, int]] = list()

        self.previous_degree = self._get_circuit_degree()
        self.previous_gate_count: int = self._get_gate_count()
        self.previous_len: float = self._len_move_to_left()

        # optimizers
        self.cancel_cnots = cnc.CancelNghCNOTs()
        self.drop_empty = cirq.optimizers.DropEmptyMoments()
        self.stick_cnots = StickCNOTs()
        self.cancel_hadamards = cnc.CancelNghHadamards()
        self.stick_multitarget = StickMultiTarget()
        self.stick_to_cnot = StickMultiTargetToCNOT()

    def _get_gate_count(self) -> int:
        counter: int = 0
        for moment in self.current_circuit:
            counter += len(moment)
        return counter

    def _apply_identity(self, action: int) -> None:
        if action == 0:
            return

        if action == 1:
            try:
                if self.could_apply_on[g.random_index][0] == CircuitIdentity.REVERSED_CNOT:
                    g.working_optimizers["rerversecnot"].optimize_circuit(self.current_circuit)
                    return

                if self.could_apply_on[g.random_index][0] == CircuitIdentity.ONE_HADAMARD_UP_LEFT:
                    g.working_optimizers["toplefth"].optimize_circuit(self.current_circuit)
                    return

                if self.could_apply_on[g.random_index][0] == CircuitIdentity.ONE_HADAMARD_LEFT_DOUBLE_RIGHT:
                    g.working_optimizers["onehleft"].optimize_circuit(self.current_circuit)
                    return

                if self.could_apply_on[g.random_index][0] == CircuitIdentity.DOUBLE_HADAMARD_LEFT_RIGHT:
                    g.working_optimizers["hadamardsquare"].optimize_circuit(self.current_circuit)

            except IndexError:
                logging.error(f'Index out of range for index {g.random_index} and list size {len(self.could_apply_on)}')

    def _optimize(self) -> float:
        add_to_reward = 0.0
        try:
            self.cancel_cnots.optimize_circuit(self.current_circuit)
            self.drop_empty.optimize_circuit(self.current_circuit)
            self.stick_cnots.optimize_circuit(self.current_circuit)
            self.cancel_hadamards.optimize_circuit(self.current_circuit)
            self.stick_multitarget.optimize_circuit(self.current_circuit)
            self.drop_empty.optimize_circuit(self.current_circuit)
            self.stick_to_cnot.optimize_circuit(self.current_circuit)
            self.drop_empty.optimize_circuit(self.current_circuit)
        except Exception as arg:
            logging.error(f'Error while applying optimization : {arg}')

        return add_to_reward

    def _get_circuit_degree(self) -> np.ndarray:
        degrees = np.zeros(len(self.current_circuit.all_qubits()))
        qubit_dict = dict()
        for qubit in self.current_circuit.all_qubits():
            qubit_dict[qubit] = len(qubit_dict)

        for moment in self.current_circuit:
            for op in moment:
                if mu.my_isinstance(op, cirq.CNOT):
                    q1 = qubit_dict[op.qubits[0]]
                    q2 = qubit_dict[op.qubits[1]]
                    degrees[q1: q1 + 1] += 1
                    degrees[q2: q2 + 1] += 1

        return np.mean(degrees)

    def _len_move_to_left(self) -> int:
        n_circuit = cirq.Circuit(self.current_circuit.all_operations(), strategy=cirq.InsertStrategy.EARLIEST)
        return len(n_circuit)

    def _break_moments(self) -> None:
        self.current_circuit = cirq.Circuit(self.current_circuit.all_operations(), strategy=cirq.InsertStrategy.NEW)

    def _exhaust_optimization(self) -> float:
        reward = 0.0
        prev_circ_repr = ""
        curr_circ_repr = circopt_utils.get_unique_representation(self.current_circuit)

        while prev_circ_repr != curr_circ_repr:
            prev_circ_repr = curr_circ_repr
            reward += self._optimize()
            curr_circ_repr = circopt_utils.get_unique_representation(self.current_circuit)

        return reward

    # @g.timing
    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        # 1. Update the environment state based on the chosen action
        action_by_value = circopt_utils.get_action_by_value(action)
        if action_by_value is not None:
            self.current_action = action_by_value

        self._apply_identity(self.current_action[2])
        reward: float = self._exhaust_optimization()

        # 2. Calculate the "reward" for the new state of the circuit
        # print('previous degree: ', self.previous_degree)

        current_degree = self._get_circuit_degree()
        current_len: int = self._len_move_to_left()
        current_gate_count: int = self._get_gate_count()

        # reward += pow((self.previous_degree - current_degree), (1 + self.previous_len / current_len))

        mexp = max(0, self.previous_len - current_len)
        contrast = (self.previous_gate_count - current_gate_count) / (self.previous_gate_count + current_gate_count)
        contrast += 2
        # reward = pow(self.previous_gate_count - current_gate_count, 1 + mexp)
        reward = pow(contrast, 1 + mexp)

        # contrast1 = (self.len_start - current_len) / (self.len_start + current_len)
        # contrast1 = 1 - (contrast1 + 1) / 2
        # contrast2 = (self.previous_len - current_len) / (
        #             self.previous_len + current_len)
        # contrast2 = 1 - (contrast2 + 2) / 2
        # reward = np.exp(contrast1)

        # reward = self.previous_len / self._len_move_to_left()

        self.previous_degree = current_degree
        self.previous_gate_count = current_gate_count
        self.previous_len = current_len

        # 3. Store the new "observation" for the state (Identity config)

        # TODO: Alexandru
        print("Reward", reward, "|C|", current_len, self.len_start, f"contrast{contrast} mexp{mexp}")
        # print(self.current_circuit)
        # input("Press any key...")

        # circuit_as_string: str = circopt_utils.get_unique_representation(self.current_circuit)
        # keep initial position of state in QTable
        # if circuit_as_string not in g.state_map_identity.keys():
        #     g.state_map_identity[circuit_as_string] = len(g.state_map_identity)
        # g.state_counter[circuit_as_string] = g.state_counter.get(circuit_as_string, 0) + 1

        # count every time a state is encountered
        # observation: int = g.state_map_identity.get(circuit_as_string)

        return 0, reward, self.done, {"current_len": current_len}

    def reset(self):
        self.current_circuit = copy.deepcopy(self.starting_circuit)
        self._exhaust_optimization()
        self.previous_degree = self._get_circuit_degree()
        g.current_moment = 0
        self.done = False
        self.could_apply_on, identity_int_string = circopt_utils.get_all_possible_identities(self.current_circuit)
        # self.could_apply_on = [i for i in self.could_apply_on if i[1] >= g.current_moment]

        if identity_int_string not in g.state_map_identity.keys():
            g.state_map_identity[identity_int_string] = len(g.state_map_identity)

        return g.state_map_identity.get(identity_int_string)

    def render(self, mode='human', close=False):
        if self.current_action == (0, 0, 0):
            print('Setting up...')
            return
        print(f'Chosen action: {self.current_action}')
        print(f'Current circuit depth: {len(self.current_circuit)}')
