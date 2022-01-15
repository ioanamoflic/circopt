import cirq
import global_stuff as g
import quantify.utils.misc_utils as mu


class StickMultiTargetToCNOT(cirq.PointOptimizer):
    def __init__(self, optimize_till: int = None):
        super().__init__()
        self.optimize_till: int = optimize_till
        self.reward = 0.0

    def optimization_at(self, circuit, index, op):
        if self.optimize_till is not None and index >= self.optimize_till:
            return None

        if len(op.qubits) >= 3:
            control_left = op.qubits[0]
            targets_left = op.qubits[1:]
            next_op_index = circuit.next_moment_operating_on([control_left], start_moment_index=index + 1)

            # if next_op_index != index + 1:
            #     return None

            if next_op_index is not None:
                cnot = circuit.operation_at(control_left, next_op_index)
                if mu.my_isinstance(cnot, cirq.CNOT):
                    control_right = cnot.qubits[0]
                    target_right = cnot.qubits[1]

                    if control_left == control_right and target_right not in targets_left:

                        prev_op_index_target = circuit.prev_moment_operating_on([target_right],
                                                                                end_moment_index=next_op_index - 1)

                        if prev_op_index_target is not None and prev_op_index_target >= index:
                            return None

                        targets = [control_left] + list(targets_left) + [target_right]
                        gate = cirq.ParallelGate(cirq.X, len(targets[1:]))
                        c_op = gate.controlled().on(*targets)
                        new_op = [c_op]

                        # print('i found multitarget and cnot to stick ', index)
                        self.reward += 0.1

                        return cirq.PointOptimizationSummary(
                            clear_span=next_op_index - index + 1,
                            clear_qubits=targets,
                            new_operations=[new_op]
                        )
        return None
