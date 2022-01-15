import cirq
from optimization.optimize_circuits import CircuitIdentity
import global_stuff as g


class HadamardSquare(cirq.PointOptimizer):
    def __init__(self, where_to: int = 0, only_count=False):
        super().__init__()
        self.where_to = where_to
        self.only_count = only_count
        self.count = 0
        self.moment_index_qubit = []

    def optimization_at(self, circuit, index, op):

        if index != self.where_to and not self.only_count:
            return None

        if g.my_isinstance(op, cirq.H):

            next_op_index = circuit.next_moment_operating_on(op.qubits, start_moment_index=index + 1)
            qubit = op.qubits[0]

            if next_op_index != index + 1:
                return None

            if next_op_index is not None:

                cnot = circuit.operation_at(qubit, next_op_index)

                if g.my_isinstance(cnot, cirq.CNOT):
                    control = cnot.qubits[0]
                    target = cnot.qubits[1]

                    if qubit == control:
                        downLeftHadamard = circuit.operation_at(target, index)

                        if g.my_isinstance(downLeftHadamard, cirq.H):

                            next_op_index = circuit.next_moment_operating_on(op.qubits, start_moment_index=index + 2)

                            if next_op_index != index + 2:
                                return None

                            if next_op_index is not None:
                                hadamard_down = circuit.operation_at(control, next_op_index)
                                hadamard_up = circuit.operation_at(target, next_op_index)

                                if g.my_isinstance(hadamard_up, cirq.H) and \
                                        g.my_isinstance(hadamard_down, cirq.H):

                                        new_op = [cirq.CNOT.on(target, control)]

                                        if self.only_count:
                                            self.count += 1
                                            self.moment_index_qubit.append((CircuitIdentity.DOUBLE_HADAMARD_LEFT_RIGHT, index, qubit))
                                            return None

                                        return cirq.PointOptimizationSummary(
                                            clear_span=next_op_index - index + 1,
                                            clear_qubits=[control, target],
                                            new_operations=[new_op]
                                        )
            return None
