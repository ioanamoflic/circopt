from typing import Tuple, List, Union, Any
import numpy as np
import gym
import cirq
import global_stuff as g
from optimization.stick_multitarget_to_CNOT import StickMultiTargetToCNOT
from optimization.stick_CNOTs import StickCNOTs
from optimization.stick_multitarget import StickMultiTarget
import circopt_utils
import quantify.optimizers as cnc
import quantify.utils.misc_utils as mu
from optimization.optimize_circuits import CircuitIdentity
import logging
import copy
import random

logging.basicConfig(filename='logfile.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s')


class CircuitEnvIdent(gym.Env):
    """
    An OpenAI Gym circuit environment.
    """
    metadata = {'render.modes': ['human']}
    NO_ACTIONS = 2
    CNOT_WEIGHT = 1.0
    MULTITARGET_WEIGHT = 10.0
    H_WEIGHT = 0.1

    def __init__(self, starting_circuit: cirq.Circuit):
        super(CircuitEnvIdent, self).__init__()
        self.current_action: Tuple[int, int, int] = (0, 0, 0)
        self.starting_circuit: cirq.Circuit = starting_circuit
        self.current_circuit: cirq.Circuit = copy.deepcopy(self.starting_circuit)
        self.done: bool = False
        self.len_start: int = self._len_move_to_left()
        self.could_apply_on: List[Tuple[int, int]] = list()

        # self.previous_degree = self._get_circuit_degree()
        # self.previous_gate_count: int = self._get_gate_count()
        # self.previous_len: float = self._len_move_to_left()

        self.max_len = 0
        self.max_gate_count = 0
        self.max_degree = 0
        self.max_weight_av = 0

        # optimizers
        self.cancel_cnots = cnc.CancelNghCNOTs()
        self.drop_empty = cirq.optimizers.DropEmptyMoments()
        self.stick_cnots = StickCNOTs()
        self.cancel_hadamards = cnc.CancelNghHadamards()
        self.stick_multitarget = StickMultiTarget()
        self.stick_to_cnot = StickMultiTargetToCNOT()

    def _get_weighted_av(self) -> float:
        div_by = self.CNOT_WEIGHT + self.MULTITARGET_WEIGHT + self.H_WEIGHT
        cnots: int = 0
        hs: int = 0
        multitargets: int = 0
        for moment in self.current_circuit:
            for op in moment:
                if len(op.qubits) > 2:
                    multitargets += 1
                elif len(op.qubits) == 2:
                    cnots += 1
                elif len(op.qubits) == 1:
                    hs += 1
        return (multitargets * self.MULTITARGET_WEIGHT + cnots * self.CNOT_WEIGHT + hs * self.H_WEIGHT) / div_by

    def _get_gate_count(self) -> int:
        counter: int = 0
        for moment in self.current_circuit:
            counter += len(moment)
        return counter

    def _get_cnot_gate_count(self) -> int:
        counter: int = 0
        for moment in self.current_circuit:
            for op in moment:
                if len(op.qubits) >= 2:
                    counter += 1
        return counter

    def _get_random_action(self) -> Tuple[Tuple[int, Any, int], int]:
        random_index = random.randint(0, len(self.could_apply_on) - 1)
        g.random_moment = self.could_apply_on[random_index][1]
        qubit = self.could_apply_on[random_index][2]
        identity = self.could_apply_on[random_index][0]

        tup = (identity, qubit, random.randint(0, 1))
        if tup not in g.action_map.keys():
            g.action_map[tup] = len(g.action_map)

        return tup, random_index

    def _apply_identity(self, action: int, index: int) -> None:
        if action == 0 or index == -1:
            return

        if action == 1:
            if self.could_apply_on[index][0] == CircuitIdentity.REVERSED_CNOT:
                g.working_optimizers["rerversecnot"].optimize_circuit(self.current_circuit)
                return

            if self.could_apply_on[index][0] == CircuitIdentity.ONE_HADAMARD_UP_LEFT:
                g.working_optimizers["toplefth"].optimize_circuit(self.current_circuit)
                return

            if self.could_apply_on[index][0] == CircuitIdentity.ONE_HADAMARD_LEFT_DOUBLE_RIGHT:
                g.working_optimizers["onehleft"].optimize_circuit(self.current_circuit)
                return

            if self.could_apply_on[index][0] == CircuitIdentity.DOUBLE_HADAMARD_LEFT_RIGHT:
                g.working_optimizers["hadamardsquare"].optimize_circuit(self.current_circuit)

    def _optimize(self) -> float:
        add_to_reward = 0.0
        try:
            self.cancel_cnots.optimize_circuit(self.current_circuit)
            add_to_reward += self.cancel_cnots.reward
            self.drop_empty.optimize_circuit(self.current_circuit)
            self.stick_cnots.optimize_circuit(self.current_circuit)
            add_to_reward += self.stick_cnots.reward
            self.cancel_hadamards.optimize_circuit(self.current_circuit)
            self.stick_multitarget.optimize_circuit(self.current_circuit)
            add_to_reward += self.stick_multitarget.reward
            self.drop_empty.optimize_circuit(self.current_circuit)
            self.stick_to_cnot.optimize_circuit(self.current_circuit)
            add_to_reward += self.stick_to_cnot.reward
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
    def step(self, action: Union[int, str]) -> Tuple[int, float, bool, dict]:
        # 1. ---------------- Update the environment state based on the chosen action ----------------

        if action == 'random':
            self.current_action, list_index = self._get_random_action()
        else:
            self.current_action = circopt_utils.get_action_by_value(action)
            list_index = [index for index, value in enumerate(self.could_apply_on)
                          if value[0] == self.current_action[0] and value[2] == self.current_action[1]]

            if len(list_index) > 0:
                list_index = list_index[0]
                g.random_moment = self.could_apply_on[list_index][1]
            else:
                list_index = -1

        print(f'Step self.could_apply_on: {self.could_apply_on}')
        print('Current circuit: \n', self.current_circuit)
        print('Chosen action: ', self.current_action)

        self._apply_identity(self.current_action[2], index=list_index)
        self._exhaust_optimization()

        print('Optimized circuit: \n', self.current_circuit)

        current_degree = self._get_circuit_degree()
        current_len: int = self._len_move_to_left()
        current_gate_count: int = self._get_gate_count()
        current_weight_av = self._get_weighted_av()

        extra = dict()
        extra["current_len"] = current_len
        extra['current_gate_count'] = current_gate_count
        extra["action"] = g.state_map_identity.get(self.current_action)

        # 2. ---------------- Calculate the "reward" for the new state of the circuit ----------------

        # mexp = max(0.0, self.previous_len - current_len)
        # contrast = (self.previous_gate_count - current_gate_count) / (self.previous_gate_count + current_gate_count)
        # contrast += 2
        # reward = pow(contrast, 1 + mexp)
        if self.max_weight_av - current_weight_av > 0:
            reward = pow(self.max_weight_av - current_weight_av,
                         1 + self.max_degree / current_degree + self.max_len / current_len)
        else:
            reward = pow(current_weight_av, 1 + self.max_degree / current_degree + self.max_len / current_len)

        print(f'Reward: {reward}')

        # 3. ---------------- Store the new "observation" for the state (Identity config) ----------------

        self.could_apply_on, identity_int_string = circopt_utils.get_all_possible_identities(self.current_circuit)
        if identity_int_string not in g.state_map_identity.keys():
            g.state_map_identity[identity_int_string] = len(g.state_map_identity)

        if len(self.could_apply_on) == 0:
            self.done = True

        observation = g.state_map_identity.get(identity_int_string)
        # self.previous_degree = current_degree
        # self.previous_gate_count = current_gate_count
        # self.previous_len = current_len

        self.max_len = max(self.max_len, current_len)
        self.max_gate_count = max(self.max_gate_count, current_gate_count)
        self.max_degree = max(self.max_degree, current_degree)

        return observation, reward, self.done, extra

    def reset(self):
        self.current_circuit = copy.deepcopy(self.starting_circuit)
        self._exhaust_optimization()
        # self.previous_degree = self._get_circuit_degree()
        # self.previous_len = self._len_move_to_left()
        # self.previous_gate_count = self._get_gate_count()
        self.done = False
        self.could_apply_on, identity_int_string = circopt_utils.get_all_possible_identities(self.current_circuit)
        if identity_int_string not in g.state_map_identity.keys():
            g.state_map_identity[identity_int_string] = len(g.state_map_identity)

        return g.state_map_identity.get(identity_int_string)

    def render(self, mode='human', close=False):
        if self.current_action == (0, 0, 0):
            print('Setting up...')
            return
        print(f'Chosen action: {self.current_action}')
        print(f'Current circuit depth: {len(self.current_circuit)}')
