import cirq
import quantify.utils.misc_utils as mu
from optimization.optimize_circuits import CircuitIdentity


class StickMultiTargetToCNOT(cirq.PointOptimizer):
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
                if mu.my_isinstance(cnot, cirq.CNOT):
                    control_right = cnot.qubits[0]
                    target_right = cnot.qubits[1]

                    if control_left == control_right and target_right not in targets_left:

                        prev_op_index_target = circuit.prev_moment_operating_on([target_right], end_moment_index=next_op_index)

                        if prev_op_index_target is not None and prev_op_index_target >= index:
                            return None

                        targets = [control_left] + list(targets_left) + [target_right]
                        gate = cirq.ParallelGate(cirq.X, len(targets[1:]))
                        c_op = gate.controlled().on(*targets)
                        new_op = [c_op]

                        if self.count_between:
                            self.count += 1
                            return None

                        if self.only_count:
                            self.count += 1
                            self.moment_index_qubit.append(
                                (CircuitIdentity.STICK_MULTITARGET_TO_CNOT.value, index, control_left)
                            )
                            return None

                        setattr(cnot, "allow", False)

                        return cirq.PointOptimizationSummary(
                            clear_span=1,
                            clear_qubits=targets,
                            new_operations=[new_op]
                        )
        return None
