import cirq
from optimization.optimize_circuits import CircuitIdentity
import quantify.utils.misc_utils as mu


class OneHLeftTwoRight(cirq.PointOptimizer):
    def __init__(self, moment=None, qubit=None, only_count=False, count_between=False):
        super().__init__()
        self.only_count = only_count
        self.count = 0
        self.moment_index_qubit = []
        self.moment = moment
        self.qubit = qubit
        self.start_moment = 0
        self.end_moment = 0
        self.count_between = count_between

    def optimization_at(self, circuit, index, op):

        if self.count_between and (index < self.start_moment or index > self.end_moment):
            return None

        if (index != self.moment or op.qubits[0] != self.qubit) and not self.only_count and not self.count_between:
            return None

        if mu.my_isinstance(op, cirq.H):

            next_op_index = circuit.next_moment_operating_on(op.qubits, start_moment_index=index + 1)
            qubit = op.qubits[0]

            if next_op_index != index + 1:
                return None

            if next_op_index is not None:

                cnot = circuit.operation_at(qubit, next_op_index)

                if mu.my_isinstance(cnot, cirq.CNOT):
                    control = cnot.qubits[0]
                    target = cnot.qubits[1]

                    if qubit == control:
                        next_op_index = circuit.next_moment_operating_on(op.qubits,
                                                                         start_moment_index=next_op_index + 1)

                        if next_op_index != index + 2:
                            return None

                        if next_op_index is not None:
                            hadamard_down = circuit.operation_at(cnot.qubits[0], next_op_index)
                            hadamard_up = circuit.operation_at(cnot.qubits[1], next_op_index)

                            if mu.my_isinstance(hadamard_up, cirq.H) and \
                                    mu.my_isinstance(hadamard_down, cirq.H):

                                if qubit == control:
                                    new_op = [cirq.H.on(target), cirq.CNOT.on(target, control)]

                                    if self.count_between:
                                        self.count += 1
                                        return None

                                    if self.only_count:
                                        self.count += 1
                                        self.moment_index_qubit.append(
                                            (CircuitIdentity.ONE_HADAMARD_LEFT_DOUBLE_RIGHT.value, index, qubit)
                                        )
                                        return None

                                    return cirq.PointOptimizationSummary(
                                        clear_span=next_op_index - index + 1,
                                        clear_qubits=[control, target],
                                        new_operations=[new_op]
                                    )
        return None
