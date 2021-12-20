import cirq


class StickMultiTarget(cirq.PointOptimizer):
    def __init__(self, optimize_till: int = None):
        super().__init__()
        self.optimize_till = optimize_till

    def optimization_at(self, circuit, index, op):
        if self.optimize_till is not None and index >= self.optimize_till:
            return None

        if len(op.qubits) >= 3:
            control_left = op.qubits[0]
            targets_left = op.qubits[1:]
            next_op_index = circuit.next_moment_operating_on(op.qubits, start_moment_index=index + 1)

            # if next_op_index != index + 1:
            #     return None

            if next_op_index is not None:
                cnot = circuit.operation_at(op.qubits[0], next_op_index)
                if cnot is not None and len(cnot.qubits) >= 3:

                    control_right = cnot.qubits[0]
                    targets_right = cnot.qubits[1:]

                    if control_left == control_right:
                        targets = [control_left] + list(targets_left) + list(targets_right)
                        gate = cirq.ParallelGate(cirq.X, len(targets[1:]))
                        c_op = gate.controlled().on(*targets)
                        new_op = [c_op]

                        return cirq.PointOptimizationSummary(
                            clear_span=next_op_index - index + 1,
                            clear_qubits=targets,
                            new_operations=[new_op]
                        )
        return   None
