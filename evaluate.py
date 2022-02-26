import copy

import circopt_utils as utils
import numpy as np
from optimization.optimize_circuits import CircuitIdentity
from circuits.ioana_random import *
import sys
import fnmatch
import os
from optimization.reverse_CNOT import ReverseCNOT
from optimization.hadamard_square import HadamardSquare
from optimization.top_left_hadamard import TopLeftHadamard
from optimization.one_H_left_2_right import OneHLeftTwoRight
from optimization.stick_multitarget_to_CNOT import StickMultiTargetToCNOT
from optimization.stick_CNOTs import StickCNOTs
from optimization.stick_multitarget import StickMultiTarget
from quantify.optimizers.cancel_ngh_cnots import CancelNghCNOTs
from quantify.optimizers.cancel_ngh_hadamard import CancelNghHadamards

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


def optimize(test_circuit, Q_Table, state_map, action_map, steps):

    for step in range(steps):
        apply_on, current_state = get_all_possible_identities(test_circuit)

        if current_state not in state_map.keys():
            print('Initial circuit state is not found in QTable :(')
            return test_circuit

        state_index = state_map[current_state]
        best_action = np.argmax(Q_Table[state_index, :])
        action = None

        for key, value in action_map.items():
            if value == best_action:
                action = key

        index = [index for index, value in enumerate(apply_on)
                 if value[0].value == action[0] and value[2].name == action[1]]

        if len(index) == 0:
            return test_circuit

        if len(index) > 0:
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

    return test_circuit


def run():
    test_or_train = sys.argv[1]
    test_number = sys.argv[2]
    steps = sys.argv[3]

    q, a, s = utils.read_train_data()
    test_circuit = None

    for file in os.listdir(f'./{test_or_train}_circuits'):
        if fnmatch.fnmatch(file, f'{test_or_train.upper()}_{test_number}.txt'):
            print(file)
            f = open(f'{test_or_train}_circuits/{test_or_train.upper()}_{test_number}.txt', 'r')
            json_string = f.read()
            test_circuit = cirq.read_json(json_text=json_string)

    test = copy.deepcopy(test_circuit)

    if test_circuit is not None:
        optimized_circuit = optimize(test, q, s, a, steps=int(steps))
        utils.plot_optimization_result(test, optimized_circuit)


if __name__ == '__main__':
    run()
