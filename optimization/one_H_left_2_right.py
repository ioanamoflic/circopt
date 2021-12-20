import cirq
from optimization.optimize_circuits import CircuitIdentity
import global_stuff as g


class OneHLeftTwoRight(cirq.PointOptimizer):
    def __init__(self, where_to: int = 0, only_count=False):
        super().__init__()
        self.where_to = where_to
        self.only_count = only_count
        self.count = 0
        self.moment_index = []

    def optimization_at(self, circuit, index, op):

        if index != self.where_to and not self.only_count:
            return None

        if isinstance(op, cirq.GateOperation) and (op.gate == cirq.H):

            next_op_index = circuit.next_moment_operating_on(op.qubits, start_moment_index=index + 1)
            qubit = op.qubits[0]

            if next_op_index != index + 1:
                return None

            if next_op_index is not None:

                cnot = circuit.operation_at(qubit, next_op_index)

                if isinstance(cnot, cirq.GateOperation) and (cnot.gate == cirq.CNOT):
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

                            if isinstance(hadamard_up, cirq.GateOperation) and (hadamard_up.gate == cirq.H) and \
                                    isinstance(hadamard_down, cirq.GateOperation) and (hadamard_down.gate == cirq.H):

                                if qubit == control:

                                    new_op = [cirq.H.on(target), cirq.CNOT.on(target, control)]

                                    if self.only_count:
                                        self.count += 1
                                        self.moment_index.append((CircuitIdentity.ONE_HADAMARD_LEFT_DOUBLE_RIGHT, index))
                                        return None

                                    return cirq.PointOptimizationSummary(
                                        clear_span=next_op_index - index + 1,
                                        clear_qubits=[control, target],
                                        new_operations=[new_op]
                                    )
        return None
