import cirq
import cirq.optimizers
from optimization.TopLeftT import TopLeftT
from optimization.TopRightT import TopRightT
from optimization.TopLeftHadamard import TopLeftHadamard
from optimization.OneHLeft2Right import OneHLeftTwoRight
from quantify.mathematics.carry_ripple_4t_adder import CarryRipple4TAdder
import routing.routing_multiple as rm
import quantify.optimizers as cnc


def top_left_t():
    # q0, q1 = cirq.LineQubit.range(2)
    # circuit = cirq.Circuit()
    # circuit.append(cirq.T.on(q0))
    # circuit.append(cirq.CNOT.on(q0, q1))
    #
    # print(circuit)
    #
    # opt_circuit = TopLeftT()
    # opt_circuit.optimize_circuit(circuit)
    #
    # print(circuit)

    circuit: cirq.Circuit = CarryRipple4TAdder(3).circuit

    circ_dec = rm.RoutingMultiple(circuit, no_decomp_sets=10, nr_bits=2)
    circ_dec.get_random_decomposition_configuration()
    decomposed_circuit = circ_dec.decompose_toffolis_in_circuit(circ_dec.configurations.pop())

    print(decomposed_circuit)
    depth_before = len(decomposed_circuit)

    for index in range(0, len(decomposed_circuit)):
        opt_circuit = TopLeftT(index)
        opt_circuit.optimize_circuit(decomposed_circuit)

        opt_circuit = TopRightT(index)
        opt_circuit.optimize_circuit(decomposed_circuit)

        opt_circuit = OneHLeftTwoRight(index)
        opt_circuit.optimize_circuit(decomposed_circuit)

        opt_circuit = TopLeftHadamard(index)
        opt_circuit.optimize_circuit(decomposed_circuit)

        cncl = cnc.CancelNghHadamards()
        cncl.optimize_circuit(decomposed_circuit)

        drop_empty = cirq.optimizers.DropEmptyMoments()
        drop_empty.optimize_circuit(decomposed_circuit)

        cncl = cnc.CancelNghCNOTs()
        cncl.optimize_circuit(decomposed_circuit)

        drop_empty = cirq.optimizers.DropEmptyMoments()
        drop_empty.optimize_circuit(decomposed_circuit)

    depth_after = len(decomposed_circuit)

    print('depth ratio: ', depth_before, depth_after)

    print(decomposed_circuit)


def top_right_t():

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    circuit.append(cirq.CNOT.on(q0, q1))
    circuit.append(cirq.T.on(q0))

    print(circuit)

    opt_circuit = TopRightT()
    opt_circuit.optimize_circuit(circuit)

    print(circuit)


def top_right_h():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on(q0))
    circuit.append(cirq.CNOT.on(q0, q1))

    print(circuit)

    opt_circuit = TopLeftHadamard()
    opt_circuit.optimize_circuit(circuit)

    print(circuit)


def top_two_right_h():

    # q0, q1, q2, q3 = cirq.LineQubit.range(4)
    # circuit = cirq.Circuit()
    # circuit.append(cirq.H.on(q1))
    # circuit.append(cirq.CNOT.on(q1, q0))
    # circuit.append([cirq.H.on(q0), cirq.H.on(q1)])
    # circuit.append(cirq.H.on(q1))
    # circuit.append(cirq.CNOT.on(q1, q0))
    # circuit.append([cirq.H.on(q0), cirq.H.on(q1)])

    circuit: cirq.Circuit = CarryRipple4TAdder(3).circuit

    circ_dec = rm.RoutingMultiple(circuit, no_decomp_sets=10, nr_bits=3)
    circ_dec.get_random_decomposition_configuration()
    decomposed_circuit = circ_dec.decompose_toffolis_in_circuit(circ_dec.configurations.pop())

    print(decomposed_circuit)

    for index in range(0, len(decomposed_circuit)):
        opt_circuit = OneHLeftTwoRight(True)
        opt_circuit.optimize_circuit(decomposed_circuit)
        print(opt_circuit.count)

    print(decomposed_circuit)



if __name__ == '__main__':
    top_left_t()