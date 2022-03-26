import copy

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


counting_moments_optimizers = {
            "onehleft": OneHLeftTwoRight(count_between=True),
            "toplefth": TopLeftHadamard(count_between=True),
            "rerversecnot": ReverseCNOT(count_between=True),
            "hadamardsquare": HadamardSquare(count_between=True),
            "cancelcnots": CancelNghCNOTs(count_between=True),
            "cancelh": CancelNghHadamards(count_between=True),
            "cnot+cnot": StickCNOTs(count_between=True),
            "multi+multi": StickMultiTarget(count_between=True),
            "multi+cnot": StickMultiTargetToCNOT(count_between=True)
}


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

drop_empty = cirq.optimizers.DropEmptyMoments()


def _get_observation(circuit, moment_range):
    observation: str = ''
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

        observation = observation + bit + '|'

    return observation


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


def optimize(test_circuit, Q_Table, state_map, action_map, steps):
    initial_circuit = copy.deepcopy(test_circuit)

    for step in range(steps):
        print(test_circuit)
        apply_on, current_state = get_all_possible_identities(test_circuit)
        # current_state = _get_observation(circuit=test_circuit, moment_range=2)

        print(f'Step {step}')
        print(f'State {current_state}')
        print(f'Gate count {_get_gate_count(circuit=test_circuit)}')
        print(f'Length: {len(cirq.Circuit(test_circuit.all_operations(), strategy=cirq.InsertStrategy.EARLIEST))}')

        if current_state not in state_map.keys():
            print('State not found in QT: ', current_state)
            utils.plot_optimization_result(initial_circuit, test_circuit)
            return test_circuit

        state_index = state_map[current_state]
        best_action = np.argmax(Q_Table[state_index, :])
        action = None

        for key, value in action_map.items():
            if value == best_action:
                action = key

        print(f'Best action index: {best_action}, best action: {action}')
        print(apply_on)

        index_list = [index for index, value in enumerate(apply_on)
                 if value[0] == action[0]
                 and value[1] // 5 == action[1]
                 and value[2].name == action[2]]

        print('Index list: ', index_list)

        for index in index_list:
            print('Match: ', apply_on[index])

        if len(index_list) == 0:
            print('No identity match found for current circuit.')
            utils.plot_optimization_result(initial_circuit, test_circuit)
            return test_circuit

        if len(index_list) > 0:
            index = index_list[0]
            identity = apply_on[index][0]
            moment = apply_on[index][1]
            qub = apply_on[index][2]

            for optimizer in working_optimizers.values():
                optimizer.moment = moment
                optimizer.qubit = qub

                if identity == CircuitIdentity.REVERSED_CNOT.value:
                    working_optimizers["rerversecnot"].optimize_circuit(test_circuit)

                elif identity == CircuitIdentity.ONE_HADAMARD_UP_LEFT.value:
                    working_optimizers["toplefth"].optimize_circuit(test_circuit)

                elif identity == CircuitIdentity.ONE_HADAMARD_LEFT_DOUBLE_RIGHT.value:
                    working_optimizers["onehleft"].optimize_circuit(test_circuit)

                elif identity == CircuitIdentity.DOUBLE_HADAMARD_LEFT_RIGHT.value:
                    working_optimizers["hadamardsquare"].optimize_circuit(test_circuit)

                elif identity == CircuitIdentity.CANCEL_CNOTS.value:
                    working_optimizers["cancelcnots"].optimize_circuit(test_circuit)

                elif identity == CircuitIdentity.CANCEL_HADAMARDS.value:
                    working_optimizers["cancelh"].optimize_circuit(test_circuit)

                elif identity == CircuitIdentity.STICK_CNOTS.value:
                    working_optimizers["cnot+cnot"].optimize_circuit(test_circuit)

                elif identity == CircuitIdentity.STICK_MULTITARGET.value:
                    working_optimizers["multi+multi"].optimize_circuit(test_circuit)

                elif identity == CircuitIdentity.STICK_MULTITARGET_TO_CNOT.value:
                    working_optimizers["multi+cnot"].optimize_circuit(test_circuit)

        drop_empty.optimize_circuit(test_circuit)

    utils.plot_optimization_result(initial_circuit, test_circuit)

    return test_circuit


def run():
    filename = sys.argv[1]
    test_or_train = sys.argv[2]
    steps = sys.argv[3]

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
        optimized_circuit = optimize(test_circuit, q, s, a, steps=int(steps))


if __name__ == '__main__':
    run()
