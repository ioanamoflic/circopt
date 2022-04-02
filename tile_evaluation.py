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


def _get_unique_representation(circuit) -> str:
    n_circuit = cirq.Circuit(circuit, strategy=cirq.InsertStrategy.EARLIEST)
    str_repr = str(n_circuit)
    return str_repr


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
        current_state = _get_unique_representation(circuit=test_circuit)

        print(f'STEP NUMBER:  {step}')
        print(f'State:')
        print(current_state)
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

        if action is None:
            return

        print(f'Best action index: {best_action}, best action: {action}')

        identity = action[0]
        moment = action[1]
        qub = action[2]

        for optimizer in working_optimizers.values():
            optimizer.moment = moment
            optimizer.qubit = cirq.NamedQubit(qub)

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
    steps = sys.argv[2]

    q, s, a = utils.read_train_data()

    print('--------------------- Results ---------------------')
    test_circuit = None

    for file in os.listdir(f'./train_circuits'):
        if fnmatch.fnmatch(file, f'{filename}'):
            print('Loading circuit from: ', file)
            f = open(f'train_circuits/{filename}', 'r')
            json_string = f.read()
            test_circuit = cirq.read_json(json_text=json_string)

    if test_circuit is not None:
        optimized_circuit = optimize(test_circuit, q, s, a, steps=int(steps))


if __name__ == '__main__':
    run()
