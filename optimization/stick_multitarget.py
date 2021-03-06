import cirq
from optimization.parallel_point_optimizer import ParallelPointOptimizer


class StickMultiTarget(ParallelPointOptimizer):
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

        if len(op.qubits) >= 3:
            control_left = op.qubits[0]
            targets_left = op.qubits[1:]
            next_op_index = circuit.next_moment_operating_on([control_left], start_moment_index=index + 1)

            if next_op_index is not None:
                cnot = circuit.operation_at(control_left, next_op_index)
                if cnot is not None and len(cnot.qubits) >= 3:

                    control_right = cnot.qubits[0]
                    targets_right = cnot.qubits[1:]

                    if control_left == control_right:
                        # de verificat sa fie targets distincti
                        sl = set(targets_left)
                        sr = set(targets_right)

                        for target in targets_right:
                            prev_op_index_target = circuit.prev_moment_operating_on([target], end_moment_index=next_op_index-1)
                            if prev_op_index_target is not None and prev_op_index_target >= index:
                                return None

                        targets = [control_left] + list(sl.union(sr).difference(sl.intersection(sr)))

                        gate = cirq.ParallelGate(cirq.X, len(targets[1:]))
                        c_op = gate.controlled().on(*targets)
                        new_op = [c_op]

                        setattr(cnot, "allow", False)

                        self.reward += 0.2
                        return cirq.PointOptimizationSummary(
                            clear_span=1,
                            clear_qubits=targets,
                            new_operations=[new_op]
                        )
        return None
