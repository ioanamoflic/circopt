import copy
from typing import List

import cirq

import circopt_utils as utils
from circuits.ioana_random import *
import sys
import fnmatch
import os
from optimization.optimize_circuits import CircuitIdentity
from optimization.reverse_CNOT import ReverseCNOT
from optimization.hadamard_square import HadamardSquare
from optimization.top_left_hadamard import TopLeftHadamard
from optimization.one_H_left_2_right import OneHLeftTwoRight
from optimization.stick_multitarget_to_CNOT import StickMultiTargetToCNOT
from optimization.stick_CNOTs import StickCNOTs
from optimization.stick_multitarget import StickMultiTarget
from quantify.optimizers.cancel_ngh_cnots import CancelNghCNOTs
from quantify.optimizers.cancel_ngh_hadamard import CancelNghHadamards
import numpy as np

working_optimizers = {
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

counting_moments_optimizers = {
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

counting_optimizers = {
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

drop_empty = cirq.optimizers.DropEmptyMoments()


def _get_observation(circuit, moment_range):
    observation_list = []
    circuit_length = len(circuit)
    i = 0

    while i < circuit_length:
        start_moment = i
        end_moment = i + moment_range

        if end_moment > circuit_length:
            end_moment = circuit_length - 1

        i = end_moment + 1

        bit = ''
        for opt_circuit in counting_moments_optimizers.values():
            opt_circuit.start_moment = start_moment
            opt_circuit.end_moment = end_moment
            opt_circuit.optimize_circuit(circuit)
            bit = bit + str(opt_circuit.count) + '_'

            opt_circuit.count = 0
            opt_circuit.moment_index_qubit.clear()

        observation_list.append(bit)

    return observation_list


def sort_tuple_list(tup):
    tup.sort(key=lambda x: x[1])
    return tup


def get_all_possible_identities(circuit):
    all_possibilities = []
    identity_state: str = ''

    for opt_circuit in counting_optimizers.values():
        opt_circuit.optimize_circuit(circuit)
        identity_state = identity_state + str(opt_circuit.count) + '_'
        all_possibilities = all_possibilities + opt_circuit.moment_index_qubit

        opt_circuit.count = 0
        opt_circuit.moment_index_qubit.clear()

    return sort_tuple_list(all_possibilities), identity_state


def _get_gate_count(circuit) -> int:
    counter: int = 0
    for moment in circuit:
        counter += len(moment)
    return counter


def optimize(test_circuit, Q_Table, state_map, action_map, moment_range, steps=10):
    initial_circuit = copy.deepcopy(test_circuit)

    for step in range(steps):

        print('Step: ', step)
        drop_empty.optimize_circuit(test_circuit)
        apply_on, _ = get_all_possible_identities(test_circuit)
        current_states = _get_observation(circuit=test_circuit, moment_range=moment_range)

        # se itereaza fiecare stare din circuit
        for i, state in enumerate(current_states):
            print(
                f'----------------------------------------------------------------------------------------------------')
            print(f'State: {state}')
            print(test_circuit)
            print(f'States for circuit: {current_states}')
            print(f'Gate count {_get_gate_count(circuit=test_circuit)}')
            print(f'Length: {len(cirq.Circuit(test_circuit.all_operations(), strategy=cirq.InsertStrategy.EARLIEST))}')
            print(f'Could apply on: {apply_on}')

            # daca starea curenta este in QTable
            if state in state_map.keys():
                state_index = state_map[state]

                # se cauta actiunea optima
                best_action = np.argmax(Q_Table[state_index, :])
                action = None
                for key, value in action_map.items():
                    if value == best_action:
                        action = key
                        break

                print(f'Best action index: {best_action}, best action: {action} for state {state}')

                # daca identitatea trebuie aplicata ( = 1)
                if action[3] == 1:
                    index_list = []

                    # se cauta toate posibilele locatii in circuit in care se poate aplica
                    for index, value in enumerate(apply_on):
                        if value[0] == action[0] and cirq.NamedQubit(action[2]) in list(value[2])\
                                and i * moment_range < value[1] < (i + 1) * moment_range:
                            index_list.append(index)

                    print('Index list: ', index_list)

                    # daca nu s-a gasit nimic ce poate face match-ul in circuitul de test, se trece la starea urmatoare
                    if len(index_list) == 0:
                        print('No identity-action match found for current circuit and state.')
                        continue

                    # altfel, se itereaza toate starile si aplica identitatile
                    for index in index_list:
                        print(f'Match si index {index}: ', apply_on[index])

                        identity: int = apply_on[index][0]
                        moment: int = apply_on[index][1]
                        qubits: List[cirq.NamedQubit] = apply_on[index][2]

                        for optimizer in working_optimizers.values():
                            optimizer.moment = moment
                            optimizer.qubits = qubits

                            if identity == CircuitIdentity.DOUBLE_HADAMARD_LEFT_RIGHT.value:
                                print(f'Optimising DOUBLE_HADAMARD_LEFT_RIGHT on moment {moment} and qubits {qubits}', )
                                working_optimizers["hadamardsquare"].optimize_circuit(test_circuit)

                            if identity == CircuitIdentity.CANCEL_CNOTS.value:
                                working_optimizers["cancelcnots"].optimize_circuit(test_circuit)

                            elif identity == CircuitIdentity.CANCEL_HADAMARDS.value:
                                working_optimizers["cancelh"].optimize_circuit(test_circuit)

            else:
                print('State not found in QTable!')

    utils.plot_optimization_result(initial_circuit, test_circuit)

    return test_circuit


def run():
    filename = sys.argv[1]
    test_or_train = sys.argv[2]
    steps = int(sys.argv[3])
    moment_range = int(sys.argv[4])

    q, s, a = utils.read_train_data()

    print('--------------------- Results ---------------------')
    test_circuit = None

    for file in os.listdir(f'./{test_or_train}_circuits'):
        if fnmatch.fnmatch(file, f'{filename}'):
            print('Loading circuit from: ', file)
            f = open(f'{test_or_train}_circuits/{filename}', 'r')
            json_string = f.read()
            test_circuit = cirq.read_json(json_text=json_string)

    if test_circuit is not None:
        optimized_circuit = optimize(test_circuit, q, s, a, moment_range=moment_range, steps=steps)


if __name__ == '__main__':
    run()
