from typing import Tuple, List, Union

import gym
import numpy as np
import cirq
import quantify.utils.misc_utils as mu
from optimization.optimize_circuits import CircuitIdentity
import logging
import copy
import random
from optimization.reverse_CNOT import ReverseCNOT
from optimization.hadamard_square import HadamardSquare
from optimization.top_left_hadamard import TopLeftHadamard
from optimization.one_H_left_2_right import OneHLeftTwoRight
from optimization.stick_multitarget_to_CNOT import StickMultiTargetToCNOT
from optimization.stick_CNOTs import StickCNOTs
from optimization.stick_multitarget import StickMultiTarget
from quantify.optimizers.cancel_ngh_cnots import CancelNghCNOTs
from quantify.optimizers.cancel_ngh_hadamard import CancelNghHadamards


logging.basicConfig(filename='logfile.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s')


class CircuitEnvIdent(gym.Env):
    """
    An OpenAI Gym circuit environment.
    """
    CNOT_WEIGHT = 5.0
    H_WEIGHT = 1.5

    def __init__(self, starting_circuit: cirq.Circuit):
        super(CircuitEnvIdent, self).__init__()
        self.current_action: Tuple[int, int, int] = (0, 0, 0)
        self.starting_circuit: cirq.Circuit = starting_circuit
        self.current_circuit: cirq.Circuit = copy.deepcopy(self.starting_circuit)
        self.done: bool = False
        self.len_start: int = self._len_move_to_left()
        self.could_apply_on: List[Tuple[int, int]] = list()
        self.max_len = self._len_move_to_left()
        self.max_gate_count = self._get_gate_count()
        self.max_degree = self._get_circuit_degree()
        self.max_weight_av = self._get_weighted_av()
        self.min_weight_av = self._get_weighted_av() / 5

        # optimizers
        self.drop_empty = cirq.optimizers.DropEmptyMoments()

    def _get_weighted_av(self) -> float:
        div_by = self.CNOT_WEIGHT + self.H_WEIGHT
        cnots: int = 0
        hs: int = 0
        for moment in self.current_circuit:
            for op in moment:
                if len(op.qubits) >= 2:
                    cnots += 1
                elif len(op.qubits) == 1:
                    hs += 1
        return (cnots * self.CNOT_WEIGHT + hs * self.H_WEIGHT) / div_by

    def _get_gate_count(self) -> int:
        counter: int = 0
        for moment in self.current_circuit:
            counter += len(moment)
        return counter

    def sort_tuple_list(self, tup):
        tup.sort(key=lambda x: x[1])
        return tup

    def _get_all_possible_identities(self):
        all_possibilities = list()
        identity_state: str = ''

        for opt_circuit in counting_optimizers.values():
            opt_circuit.optimize_circuit(self.current_circuit)
            identity_state = identity_state + str(opt_circuit.count) + '_'
            all_possibilities = all_possibilities + opt_circuit.moment_index_qubit

            opt_circuit.count = 0
            opt_circuit.moment_index_qubit.clear()

        return self.sort_tuple_list(all_possibilities), identity_state

    def _apply_identity(self, action: int, index: int):
        if action == 0 or index == -1:
            return

        if action == 1:
            identity = self.could_apply_on[index][0]
            moment = self.could_apply_on[index][1]
            qub = self.could_apply_on[index][2]

            for optimizer in working_optimizers.values():
                optimizer.moment = moment
                optimizer.qubit = qub

            try:
                if identity == CircuitIdentity.REVERSED_CNOT:
                    working_optimizers["rerversecnot"].optimize_circuit(self.current_circuit)
                    return

                if identity == CircuitIdentity.ONE_HADAMARD_UP_LEFT:
                    working_optimizers["toplefth"].optimize_circuit(self.current_circuit)
                    return

                if identity == CircuitIdentity.ONE_HADAMARD_LEFT_DOUBLE_RIGHT:
                    working_optimizers["onehleft"].optimize_circuit(self.current_circuit)
                    return

                if identity == CircuitIdentity.DOUBLE_HADAMARD_LEFT_RIGHT:
                    working_optimizers["hadamardsquare"].optimize_circuit(self.current_circuit)
                    return

                if identity == CircuitIdentity.CANCEL_CNOTS:
                    working_optimizers["cancelcnots"].optimize_circuit(self.current_circuit)
                    return

                if identity == CircuitIdentity.CANCEL_HADAMARDS:
                    working_optimizers["cancelh"].optimize_circuit(self.current_circuit)
                    return

                if identity == CircuitIdentity.STICK_CNOTS:
                    working_optimizers["cnot+cnot"].optimize_circuit(self.current_circuit)
                    return

                if identity == CircuitIdentity.STICK_MULTITARGET:
                    working_optimizers["multi+multi"].optimize_circuit(self.current_circuit)
                    return

                if identity == CircuitIdentity.STICK_MULTITARGET_TO_CNOT:
                    working_optimizers["multi+cnot"].optimize_circuit(self.current_circuit)

            except Exception as e:
                logging.error(f'Error during optimization! {e}')

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

    def step(self, action: Union[int, str]) -> Tuple[str, float, bool, dict]:
        # 1. ---------------- Update the environment state based on the chosen action ----------------

        print('Current circuit: \n', self.current_circuit)

        if action == 'random':
            list_index = random.randint(0, len(self.could_apply_on) - 1)
            qubit = self.could_apply_on[list_index][2]
            identity = self.could_apply_on[list_index][0]
            self.current_action = (identity, qubit, random.randint(0, 1))
        else:
            self.current_action = action
            list_index = [index for index, value in enumerate(self.could_apply_on)
                          if value[0] == self.current_action[0] and value[2].name == self.current_action[1]]

            list_index = list_index[0] if len(list_index) > 0 else -1

        self._apply_identity(self.current_action[2], index=list_index)
        self.drop_empty.optimize_circuit(self.current_circuit)
        print('Optimized circuit: \n', self.current_circuit)

        current_degree = self._get_circuit_degree()
        current_len: int = self._len_move_to_left()
        current_gate_count: int = self._get_gate_count()
        current_weight_av = self._get_weighted_av()

        info = dict()
        info["current_len"] = current_len
        info['current_gate_count'] = current_gate_count
        info["action"] = self.current_action

        # 2. ---------------- Calculate the "reward" for the new state of the circuit ----------------

        reward = np.exp((1 + (self.max_degree / current_degree) * (self.max_len / current_len))
                        * np.log(1 + self.min_weight_av / current_weight_av))

        print(f'Reward: {reward} \n'
              f'Max degree: {self.max_degree} \n'
              f'Current degree: {current_degree} \n'
              f'Max len: {self.max_len} \n'
              f'Current len: {current_len} \n'
              f'Min w av: {self.min_weight_av} \n'
              f'Current w av: {current_weight_av} \n')

        self.max_len = max(self.max_len, current_len)
        self.max_gate_count = max(self.max_gate_count, current_gate_count)
        self.max_degree = max(self.max_degree, current_degree)
        self.max_weight_av = max(self.max_weight_av, current_weight_av)
        self.min_weight_av = min(self.min_weight_av, current_weight_av)

        # 3. ---------------- Store the new "observation" for the state (Identity config) ----------------

        self.could_apply_on, identity_int_string = self._get_all_possible_identities()

        if len(self.could_apply_on) == 0:
            self.done = True

        observation = identity_int_string
        info["state"] = observation

        return observation, reward, self.done, info

    def reset(self):
        self.current_circuit = copy.deepcopy(self.starting_circuit)
        self.done = False
        self.could_apply_on, identity_int_string = self._get_all_possible_identities()
        return identity_int_string

    def render(self, mode='human', close=False):
        if self.current_action == (0, 0, 0):
            print('Setting up...')
            return
        print(f'Chosen action: {self.current_action}')
        print(f'Current circuit depth: {len(self.current_circuit)}')


counting_optimizers = {
    "onehleft": OneHLeftTwoRight(only_count=True),
    "toplefth": TopLeftHadamard(only_count=True),
    "rerversecnot": ReverseCNOT(only_count=True),
    "hadamardsquare": HadamardSquare(only_count=True),
    "cancelcnots": CancelNghCNOTs(only_count=True),
    "cancelh": CancelNghHadamards(only_count=True),
    "cnot+cnot": StickCNOTs(only_count=True),
    "multi+multi": StickMultiTarget(only_count=True),
    "multi+cnot": StickMultiTargetToCNOT(only_count=True)
}

working_optimizers = {
    "onehleft": OneHLeftTwoRight(),
    "toplefth": TopLeftHadamard(),
    "rerversecnot": ReverseCNOT(),
    "hadamardsquare": HadamardSquare(),
    "cancelcnots": CancelNghCNOTs(),
    "cancelh": CancelNghHadamards(),
    "cnot+cnot": StickCNOTs(),
    "multi+multi": StickMultiTarget(),
    "multi+cnot": StickMultiTargetToCNOT()
}