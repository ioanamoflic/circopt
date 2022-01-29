import cirq
import quantify.utils.misc_utils as mu
from optimization.parallel_point_optimizer import ParallelPointOptimizer


class StickCNOTs(ParallelPointOptimizer):
    def __init__(self, optimize_till: int = None):
        super().__init__()
        self.optimize_till = optimize_till
        self.reward = 0.0

    def optimization_at(self, circuit, index, op):
        if self.optimize_till is not None and index >= self.optimize_till:
            return None

        if hasattr(op, "allow"):
            return cirq.PointOptimizationSummary(
                clear_span=1,
                clear_qubits=op.qubits,
                new_operations=[]
            )

        if mu.my_isinstance(op, cirq.CNOT):
            next_op_index = circuit.next_moment_operating_on([op.qubits[0]], start_moment_index=index + 1)
            qubit = op.qubits[0]

            if next_op_index is not None:
                cnot = circuit.operation_at(qubit, next_op_index)
                if mu.my_isinstance(cnot, cirq.CNOT):
                    control = cnot.qubits[0]
                    target = cnot.qubits[1]
                    if qubit == control:
                        prev_op_index_target = circuit.prev_moment_operating_on([target],
                                                                                end_moment_index=next_op_index - 1)

                        if prev_op_index_target is not None and prev_op_index_target >= index:
                            return None

                        targets = [qubit, op.qubits[1], target]

                        if len(set(targets)) < len(targets):
                            return None

                        gate = cirq.ParallelGate(cirq.X, len(targets[1:]))
                        c_op = gate.controlled().on(*targets)

                        new_op = [c_op]

                        self.reward += 0.1

                        # remove remaining op (cnot)
                        setattr(cnot, "allow", False)

                        return cirq.PointOptimizationSummary(
                            clear_span=1,
                            clear_qubits=targets,
                            new_operations=[new_op]
                        )
        return None
