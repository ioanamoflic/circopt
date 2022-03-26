import random
import numpy as np
import cirq
from cirq import InsertStrategy

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


def add_random_CNOT(circuit: cirq.Circuit, qubits):
    control = qubits[random.randint(0, len(qubits) - 1)]
    target = qubits[random.randint(0, len(qubits) - 1)]

    while control == target:
        target = qubits[random.randint(0, len(qubits) - 1)]

    circuit.append([cirq.CNOT.on(control, target)], strategy=cirq.InsertStrategy.NEW)
    return circuit


def add_random_H(circuit: cirq.Circuit, qubits):
    circuit.append([cirq.H.on(qubits[random.randint(0, len(qubits) - 1)])], strategy=cirq.InsertStrategy.NEW)

    return circuit


def add_random_H_pair(circuit: cirq.Circuit, qubits):
    circuit.append([cirq.H.on(qubits[random.randint(0, len(qubits) - 1)])], strategy=cirq.InsertStrategy.NEW)
    circuit.append([cirq.H.on(qubits[random.randint(0, len(qubits) - 1)])], strategy=cirq.InsertStrategy.NEW)

    return circuit


def add_random_CNOT_pair(circuit: cirq.Circuit, qubits):
    control = qubits[random.randint(0, len(qubits) - 1)]
    target = qubits[random.randint(0, len(qubits) - 1)]

    while control == target:
        target = qubits[random.randint(0, len(qubits) - 1)]

    circuit.append([cirq.CNOT.on(control, target)], strategy=cirq.InsertStrategy.NEW)
    circuit.append([cirq.CNOT.on(control, target)], strategy=cirq.InsertStrategy.NEW)

    return circuit


def get_random_circuit(nr_qubits: int, big_o_const: int):
    # TODO: Use maybe https://quantumai.google/reference/python/cirq/testing/random_circuit

    qubits = [cirq.NamedQubit(str(i)) for i in range(nr_qubits)]
    circuit = cirq.Circuit()

    degree = 2
    total_depth = big_o_const * nr_qubits ** degree
    for i in range(total_depth):
        if random.randint(1, 10) <= 4:
            circuit = add_random_CNOT(circuit, qubits)
        else:
            circuit = add_random_H(circuit, qubits)

    return circuit


def get_dummy_circuit(nr_qubits: int, big_o_const: int):
    qubits = [cirq.NamedQubit(str(i)) for i in range(nr_qubits)]
    circuit = cirq.Circuit()

    degree = 2
    total_depth = big_o_const * nr_qubits
    for i in range(total_depth):
        if random.randint(1, 10) <= 4:
            circuit = add_random_CNOT_pair(circuit, qubits)
        elif random.randint(1, 10) == 5 or random.randint(1, 10) == 6:
            circuit = add_random_H(circuit, qubits)
            circuit = add_random_CNOT(circuit, qubits)
        else:
            circuit = add_random_H_pair(circuit, qubits)

    return circuit


def get_test_circuit():
    circuit = cirq.Circuit()
    q0, q1, q2, q3, q4 = [cirq.NamedQubit(str(i)) for i in range(5)]
    circuit.append([cirq.CNOT.on(q4, q2)], strategy=InsertStrategy.NEW)
    circuit.append([cirq.H.on(q0)], strategy=InsertStrategy.NEW)
    circuit.append([cirq.H.on(q2)], strategy=InsertStrategy.NEW)
    circuit.append([cirq.H.on(q0)], strategy=InsertStrategy.NEW)
    circuit.append([cirq.H.on(q4)], strategy=InsertStrategy.NEW)
    circuit.append([cirq.H.on(q0)], strategy=InsertStrategy.NEW)
    circuit.append([cirq.CNOT.on(q1, q0)], strategy=InsertStrategy.NEW)
    circuit.append([cirq.H.on(q4)], strategy=InsertStrategy.NEW)
    circuit.append([cirq.H.on(q4)], strategy=InsertStrategy.NEW)
    circuit.append([cirq.CNOT.on(q2, q3)], strategy=InsertStrategy.NEW)
    circuit.append([cirq.H.on(q3)], strategy=InsertStrategy.NEW)
    circuit.append([cirq.H.on(q1)], strategy=InsertStrategy.NEW)
    circuit.append([cirq.CNOT.on(q4, q0)], strategy=InsertStrategy.NEW)
    circuit.append([cirq.H.on(q4)], strategy=InsertStrategy.NEW)
    circuit.append([cirq.CNOT.on(q4, q0)], strategy=InsertStrategy.NEW)

    return circuit


def save_starting_circuit():
    qubits = [cirq.NamedQubit(str(i)) for i in range(10)]
    circuit = cirq.Circuit()

    total_depth = 30
    for i in range(total_depth):
        if random.randint(1, 10) <= 4:
            circuit = add_random_CNOT(circuit, qubits)
        else:
            circuit = add_random_H(circuit, qubits)

    f = open(f'train_circuits/start.txt', 'w')
    json_string = cirq.to_json(circuit)
    f.write(json_string)
    f.close()

    return circuit


def get_starting_circuit():
    f = open(f'train_circuits/start.txt', 'r')
    json = f.read()
    f.close()
    circuit = cirq.read_json(json_text=json)
    return circuit


def generate_start_bench():
    circuit = get_starting_circuit()
    last_stop = 0

    for i in range(8):

        if last_stop >= len(circuit):
            last_stop = 0

        could_apply_on, _ = get_all_possible_identities(circuit)
        for j in range(len(could_apply_on)):

            if could_apply_on[j][0] == CircuitIdentity.REVERSED_CNOT.value and could_apply_on[j][1] >= last_stop:
                last_stop = could_apply_on[j][1] + 2
                working_optimizers["rerversecnot"].moment = could_apply_on[j][1]
                working_optimizers["rerversecnot"].qubit = could_apply_on[j][2]
                working_optimizers["rerversecnot"].optimize_circuit(circuit)
                f = open(f'train_circuits/START_0{i}.txt', 'w')
                json_string = cirq.to_json(circuit)
                f.write(json_string)
                f.close()
                break

            if could_apply_on[j][0] == CircuitIdentity.ONE_HADAMARD_UP_LEFT.value and could_apply_on[j][1] >= last_stop:
                last_stop = could_apply_on[j][1] + 2
                working_optimizers["toplefth"].moment = could_apply_on[j][1]
                working_optimizers["toplefth"].qubit = could_apply_on[j][2]
                working_optimizers["toplefth"].optimize_circuit(circuit)
                f = open(f'train_circuits/START_0{i}.txt', 'w')
                json_string = cirq.to_json(circuit)
                f.write(json_string)
                f.close()
                break

    return circuit


def generate_test_and_train():
    for i in range(30):
        # CK_unopt = get_random_circuit(nr_qubits=10, big_o_const=i//10+1)
        CK = get_dummy_circuit(nr_qubits=10, big_o_const=i//10+1)
        # ? CK_opt = circopt_utils.exhaust_optimization(CK_unopt)
        # CK_opt = apply_random_identities(CK_unopt)

        if 6 < i % 10 < 10:
            f = open(f'test_circuits/BASIC_TEST_{i}.txt', 'w')
        else:
            f = open(f'train_circuits/BASIC_TRAIN_{i}.txt', 'w')

        json_string = cirq.to_json(CK)
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