import cirq

from optimization.optimize_circuits import CircuitIdentity
from circuits.ioana_random import *
import circopt_utils
from optimization.reverse_CNOT import ReverseCNOT
from optimization.hadamard_square import HadamardSquare
from optimization.top_left_hadamard import TopLeftHadamard
from optimization.one_H_left_2_right import OneHLeftTwoRight
from optimization.stick_multitarget_to_CNOT import StickMultiTargetToCNOT
from optimization.stick_CNOTs import StickCNOTs
from optimization.stick_multitarget import StickMultiTarget
from quantify.optimizers.cancel_ngh_cnots import CancelNghCNOTs
from quantify.optimizers.cancel_ngh_hadamard import CancelNghHadamards

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


def generate():
    for i in range(100):
        CK_unopt = get_random_circuit(nr_qubits=10, big_o_const=i//10+1)
        CK_opt = circopt_utils.exhaust_optimization(CK_unopt)
        CK_opt = apply_random_identities(CK_opt)

        if 6 < i % 10 < 10:
            f = open(f'test_circuits/TEST_{i}.txt', 'w')
        else:
            f = open(f'train_circuits/TRAIN_{i}.txt', 'w')

        json_string = cirq.to_json(CK_opt)
        f.write(json_string)
        f.close()


def apply_random_identities(circuit: cirq.Circuit):
    gate_count = len(list(circuit.all_operations()))
    apply_for = random.randint(gate_count // 5, gate_count // 3)

    for i in range(apply_for):
        could_apply_on, _ = get_all_possible_identities(circuit)
        index = random.randint(0, len(could_apply_on) - 1)

        for optimizer in working_optimizers.values():
            optimizer.moment = could_apply_on[index][1]
            optimizer.qubit = could_apply_on[index][2]

        if could_apply_on[index][0] == CircuitIdentity.ONE_HADAMARD_UP_LEFT:
            working_optimizers["toplefth"].optimize_circuit(circuit)

        if could_apply_on[index][0] == CircuitIdentity.REVERSED_CNOT and could_apply_on[index - 1][
            0] != CircuitIdentity.DOUBLE_HADAMARD_LEFT_RIGHT:
            working_optimizers["rerversecnot"].optimize_circuit(circuit)

    return circuit


def get_all_possible_identities(circuit):
    all_possibilities = list()
    identity_state: str = ''

    for opt_circuit in counting_optimizers.values():
        opt_circuit.optimize_circuit(circuit)
        identity_state = identity_state + str(opt_circuit.count) + '_'
        all_possibilities = all_possibilities + opt_circuit.moment_index_qubit

        opt_circuit.count = 0
        opt_circuit.moment_index_qubit.clear()

    return sort_tuple_list(all_possibilities), identity_state


def sort_tuple_list(tup):
    tup.sort(key=lambda x: x[1])
    return tup

