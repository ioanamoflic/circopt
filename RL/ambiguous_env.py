import math
from typing import Tuple, List, Union

import gym
import numpy as np
import cirq
import quantify.utils.misc_utils as mu
from optimization.optimize_circuits import CircuitIdentity
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
from logging import getLogger, INFO
from concurrent_log_handler import ConcurrentRotatingFileHandler
import os

log = getLogger(__name__)
logfile = os.path.abspath("./logfile.log")
rotateHandler = ConcurrentRotatingFileHandler(logfile, "a")
log.addHandler(rotateHandler)
log.setLevel(INFO)


class AmbiguousEnv(gym.Env):
    """
    An OpenAI Gym circuit environment.
    """
    CNOT_WEIGHT = 5.0
    H_WEIGHT = 1.5

    def __init__(self, starting_circuit: cirq.Circuit, circuit_name, moment_range=1):
        super(AmbiguousEnv, self).__init__()
        self.ep = 0
        self.circuit_name = circuit_name
        self.moment_range = moment_range
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

        self.counting_optimizers = {
            # "onehleft": OneHLeftTwoRight(only_count=True),
            # "toplefth": TopLeftHadamard(only_count=True),
            # "rerversecnot": ReverseCNOT(only_count=True),
            "hadamardsquare": HadamardSquare(only_count=True),
            "cancelcnots": CancelNghCNOTs(only_count=True),
            "cancelh": CancelNghHadamards(only_count=True),
            # "cnot+cnot": StickCNOTs(only_count=True),
            # "multi+multi": StickMultiTarget(only_count=True),
            # "multi+cnot": StickMultiTargetToCNOT(only_count=True)
        }

        self.counting_moments_optimizers = {
            # "onehleft": OneHLeftTwoRight(count_between=True),
            # "toplefth": TopLeftHadamard(count_between=True),
            # "rerversecnot": ReverseCNOT(count_between=True),
            "hadamardsquare": HadamardSquare(count_between=True),
            "cancelcnots": CancelNghCNOTs(count_between=True),
            "cancelh": CancelNghHadamards(count_between=True),
            # "cnot+cnot": StickCNOTs(count_between=True),
            # "multi+multi": StickMultiTarget(count_between=True),
            # "multi+cnot": StickMultiTargetToCNOT(count_between=True)
        }

        self.working_optimizers = {
            # "onehleft": OneHLeftTwoRight(),
            # "toplefth": TopLeftHadamard(),
            # "rerversecnot": ReverseCNOT(),
            "hadamardsquare": HadamardSquare(),
            "cancelcnots": CancelNghCNOTs(),
            "cancelh": CancelNghHadamards(),
            # "cnot+cnot": StickCNOTs(),
            # "multi+multi": StickMultiTarget(),
            # "multi+cnot": StickMultiTargetToCNOT()
        }

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

        for opt_circuit in self.counting_optimizers.values():
            opt_circuit.optimize_circuit(self.current_circuit)
            identity_state = identity_state + str(opt_circuit.count) + '_'
            all_possibilities = all_possibilities + opt_circuit.moment_index_qubit

            opt_circuit.count = 0
            opt_circuit.moment_index_qubit.clear()

        return self.sort_tuple_list(all_possibilities), identity_state

    # def _get_observation(self):
    #     observation: str = ''
    #     circuit_length = len(self.current_circuit)
    #     i = 0
    #
    #     while i < circuit_length:
    #         start_moment = i
    #         end_moment = i + self.moment_range
    #
    #         if end_moment > circuit_length:
    #             end_moment = circuit_length - 1
    #
    #         i = end_moment + 1
    #
    #         bit = ''
    #         for opt_circuit in self.counting_moments_optimizers.values():
    #             opt_circuit.start_moment = start_moment
    #             opt_circuit.end_moment = end_moment
    #             opt_circuit.optimize_circuit(self.current_circuit)
    #             bit = bit + str(opt_circuit.count) + '_'
    #
    #             opt_circuit.count = 0
    #             opt_circuit.moment_index_qubit.clear()
    #
    #         observation = observation + bit + '|'
    #
    #     return observation

    # def _get_observation(self):
    #     bit = ''
    #     for opt_circuit in self.counting_optimizers.values():
    #         opt_circuit.optimize_circuit(self.current_circuit)
    #         bit = bit + str(opt_circuit.count) + '_'
    #
    #         opt_circuit.count = 0
    #         opt_circuit.moment_index_qubit.clear()
    #     return bit

    def _apply_identity(self, index: int):
        if index == -1:
            return

        identity = self.could_apply_on[index][0]
        moment = self.could_apply_on[index][1]
        qub = self.could_apply_on[index][2]

        for optimizer in self.working_optimizers.values():
            optimizer.moment = moment
            optimizer.qubit = qub

        try:
            if identity == CircuitIdentity.DOUBLE_HADAMARD_LEFT_RIGHT.value:
                self.working_optimizers["hadamardsquare"].optimize_circuit(self.current_circuit)
                return

            if identity == CircuitIdentity.CANCEL_CNOTS.value:
                self.working_optimizers["cancelcnots"].optimize_circuit(self.current_circuit)
                return

            if identity == CircuitIdentity.CANCEL_HADAMARDS.value:
                self.working_optimizers["cancelh"].optimize_circuit(self.current_circuit)
                return

                # if identity == CircuitIdentity.REVERSED_CNOT.value:
                #     self.working_optimizers["rerversecnot"].optimize_circuit(self.current_circuit)
                #     return

                # if identity == CircuitIdentity.ONE_HADAMARD_UP_LEFT.value:
                #     self.working_optimizers["toplefth"].optimize_circuit(self.current_circuit)
                #     return
                #
                # if identity == CircuitIdentity.ONE_HADAMARD_LEFT_DOUBLE_RIGHT.value:
                #     self.working_optimizers["onehleft"].optimize_circuit(self.current_circuit)
                #     return
                #
                # if identity == CircuitIdentity.STICK_CNOTS.value:
                #     self.working_optimizers["cnot+cnot"].optimize_circuit(self.current_circuit)
                #     return
                #
                # if identity == CircuitIdentity.STICK_MULTITARGET.value:
                #     self.working_optimizers["multi+multi"].optimize_circuit(self.current_circuit)
                #     return

                # if identity == CircuitIdentity.STICK_MULTITARGET_TO_CNOT.value:
                #     self.working_optimizers["multi+cnot"].optimize_circuit(self.current_circuit)

        except Exception as e:
            log.info(f'Error during optimization! {e}')

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
        if action == 'random':
            where = random.randint(0, len(self.could_apply_on) - 1)
            identity = self.could_apply_on[where][0]
            moment = self.could_apply_on[where][1]
            qubit = self.could_apply_on[where][2]
            self.current_action = (identity, moment // self.moment_range, qubit.name, random.randint(0, 1))
        else:
            list_index = [index for index, apply_on in enumerate(self.could_apply_on)
                          if apply_on[0] == action[0]
                          and apply_on[1] // self.moment_range == action[1]
                          and apply_on[2].name == action[2]
                          ]
            self.current_action = action

            where = list_index[random.randint(0, len(list_index) - 1)] if len(list_index) > 0 else -1

        # print('Chosen action: ', self.current_action)
        # print('Circuit before:\n', self.current_circuit)

        if self.current_action[3] == 1:
            self._apply_identity(index=where)
            self.drop_empty.optimize_circuit(self.current_circuit)

        if len(self.current_circuit) == 0:
            return "", 100, True, {"action": self.current_action, "state": "", "current_len": 0,
                                   "current_gate_count": 0}

        # print('Circuit after:')
        # print(self.current_circuit)

        current_degree = self._get_circuit_degree()
        current_len: int = self._len_move_to_left()
        current_gate_count: int = self._get_gate_count()
        current_weight_av = self._get_weighted_av()

        info = dict()
        info["current_len"] = current_len
        info['current_gate_count'] = current_gate_count
        info["action"] = self.current_action

        # 2. ---------------- Calculate the "reward" for the new state of the circuit ----------------
        reward = 100
        if current_degree * current_len * current_weight_av != 0:
            reward = np.exp((1 + (self.max_degree / current_degree) * (self.max_len / current_len))
                            * np.log(1 + self.min_weight_av / current_weight_av))

        if math.isinf(reward) or math.isnan(reward):
            reward = 100

        # print('Reward: ', reward)

        # e, reward, max_degree, current_degree, max_len, current_len, min_w_av, current_w_av
        log.info(
            f'{self.ep},{reward},{self.max_degree},{current_degree},{self.max_len},{current_len},{self.min_weight_av},{current_weight_av},{self.circuit_name}')

        self.max_len = max(self.max_len, current_len)
        self.max_gate_count = max(self.max_gate_count, current_gate_count)
        self.max_degree = max(self.max_degree, current_degree)
        self.max_weight_av = max(self.max_weight_av, current_weight_av)
        self.min_weight_av = min(self.min_weight_av, current_weight_av)

        # 3. ---------------- Store the new "observation" for the state (Identity config) ----------------

        self.could_apply_on, observation = self._get_all_possible_identities()
        #print(self.could_apply_on)
        #observation = self._get_observation()
        info["state"] = observation

        if len(self.could_apply_on) == 0:
            self.done = True

        # print('State: ', observation)

        return observation, reward, self.done, info

    def reset(self):
        self.ep += 1
        self.current_circuit = copy.deepcopy(self.starting_circuit)
        self.done = False
        self.could_apply_on, identity_int_string = self._get_all_possible_identities()
        # self.could_apply_on = self._get_all_possible_identities()

        # return self._get_observation()
        return identity_int_string

    def render(self, mode='human', close=False):
        if self.current_action == (0, 0, 0):
            print('Setting up...')
            return
        print(f'Chosen action: {self.current_action}')
        print(f'Current circuit depth: {len(self.current_circuit)}')
