import cirq
from optimization.optimize_circuits import CircuitIdentity
from optimization.parallel_point_optimizer import ParallelPointOptimizer

import global_stuff as g
import quantify.utils.misc_utils as mu


class HadamardSquare(ParallelPointOptimizer):
    def __init__(self, only_count=False):
        super().__init__(only_count=only_count)

    def optimization_at(self, circuit, index, op):

        if index != g.random_moment and not self.only_count:
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
                        downLeftHadamard = circuit.operation_at(target, index)

                        if mu.my_isinstance(downLeftHadamard, cirq.H):

                            next_op_index = circuit.next_moment_operating_on(op.qubits, start_moment_index=index + 2)

                            if next_op_index != index + 2:
                                return None

                            if next_op_index is not None:
                                hadamard_down = circuit.operation_at(control, next_op_index)
                                hadamard_up = circuit.operation_at(target, next_op_index)

                                if mu.my_isinstance(hadamard_up, cirq.H) and \
                                        mu.my_isinstance(hadamard_down, cirq.H):

                                        new_op = [cirq.CNOT.on(target, control)]

                                        if self.only_count:
                                            self.increase_opt_counter(CircuitIdentity.DOUBLE_HADAMARD_LEFT_RIGHT, index, qubit)
                                            return None

                                        return cirq.PointOptimizationSummary(
                                            clear_span=next_op_index - index + 1,
                                            clear_qubits=[control, target],
                                            new_operations=[new_op]
                                        )
            return None
