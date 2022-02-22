import cirq

from optimization.optimize_circuits import CircuitIdentity


class StickMultiTarget(cirq.PointOptimizer):
    def __init__(self, moment=None, qubit=None, only_count=False):
        super().__init__()
        self.only_count = only_count
        self.count = 0
        self.moment_index_qubit = []
        self.moment = moment
        self.qubit = qubit

    def optimization_at(self, circuit, index, op):
        if (index != self.moment or op.qubits[0] != self.qubit) and not self.only_count:
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

                        if self.only_count:
                            self.count += 1
                            self.moment_index_qubit.append(
                                (CircuitIdentity.STICK_MULTITARGET.value, index, op.qubits[0])
                            )
                            return None

                        setattr(cnot, "allow", False)

                        return cirq.PointOptimizationSummary(
                            clear_span=1,
                            clear_qubits=targets,
                            new_operations=[new_op]
                        )
        return None
