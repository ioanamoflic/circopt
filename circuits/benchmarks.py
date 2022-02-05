import cirq
import random

from optimization.optimize_circuits import CircuitIdentity
from optimization.stick_CNOTs import StickCNOTs
from optimization.stick_multitarget import StickMultiTarget
from optimization.stick_multitarget_to_CNOT import StickMultiTargetToCNOT
from circuits.ioana_random import *
import circopt_utils
import quantify.optimizers as cnc
import global_stuff as globals

cancel_cnots = cnc.CancelNghCNOTs()
drop_empty = cirq.optimizers.DropEmptyMoments()
stick_cnots = StickCNOTs()
cancel_hadamards = cnc.CancelNghHadamards()
stick_multitarget = StickMultiTarget()
stick_to_cnot = StickMultiTargetToCNOT()


def generate():
    circuit_height = [3, 3, 3, 3, 3]
    big_o = [1, 2, 3, 3, 3]

    for i in range(min(len(big_o), len(circuit_height))):
        CK_unopt = get_random_circuit(circuit_height[i], big_o[i])
        CK_opt = exhaust_optimization(CK_unopt)
        print(CK_opt)
        print('\n')

        CK_opt = apply_random_identities(CK_opt)
        print(CK_opt)
        print('\n')
        print('\n')
        print('\n')
        f = open(f'CR_{i}.txt', 'w')

        json_string = cirq.to_json(CK_opt)
        f.write(json_string)
        f.close()


def optimize(circuit):
    cancel_cnots.optimize_circuit(circuit)
    drop_empty.optimize_circuit(circuit)
    stick_cnots.optimize_circuit(circuit)
    cancel_hadamards.optimize_circuit(circuit)
    stick_multitarget.optimize_circuit(circuit)
    drop_empty.optimize_circuit(circuit)
    stick_to_cnot.optimize_circuit(circuit)
    drop_empty.optimize_circuit(circuit)

    return circuit


def exhaust_optimization(circuit):
    prev_circ_repr: str = ""
    curr_circ_repr: str = circopt_utils.get_unique_representation(circuit)

    while prev_circ_repr != curr_circ_repr:
        prev_circ_repr = curr_circ_repr
        circuit = optimize(circuit)
        curr_circ_repr = circopt_utils.get_unique_representation(circuit)

    return circuit


def apply_random_identities(circuit):
    apply_for = 30

    for i in range(apply_for):
        could_apply_on, _ = circopt_utils.get_all_possible_identities(circuit)
        random_index = random.randint(0, len(could_apply_on) - 1)
        globals.random_moment = random_index

        if could_apply_on[random_index][0] == CircuitIdentity.ONE_HADAMARD_UP_LEFT:
            globals.working_optimizers["toplefth"].optimize_circuit(circuit)

        if could_apply_on[random_index][0] == CircuitIdentity.REVERSED_CNOT and could_apply_on[random_index - 1][0] != CircuitIdentity.DOUBLE_HADAMARD_LEFT_RIGHT:
            globals.working_optimizers["rerversecnot"].optimize_circuit(circuit)

    return circuit
