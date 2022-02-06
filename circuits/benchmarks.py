from optimization.optimize_circuits import CircuitIdentity
from circuits.ioana_random import *
import circopt_utils
import global_stuff as globals


def generate():
    circuit_height = [3, 3, 3, 3, 3]
    big_o = [1, 2, 3, 3, 3]

    for i in range(min(len(big_o), len(circuit_height))):
        CK_unopt = get_random_circuit(circuit_height[i], big_o[i])
        CK_opt = circopt_utils.exhaust_optimization(CK_unopt)
        CK_opt = apply_random_identities(CK_opt)
        f = open(f'CR_{i}.txt', 'w')

        json_string = cirq.to_json(CK_opt)
        f.write(json_string)
        f.close()


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
