from collections import Iterable

import cirq
from optimization.optimize_circuits import CircuitIdentity


class StickMultiTarget(cirq.PointOptimizer):
    def __init__(self, optimize_til: int = None):
        super().__init__()
        self.optimize_til = optimize_til

    def optimization_at(self, circuit, index, op):

        if self.optimize_til is not None and index >= self.optimize_til:
            return None

        if len(op.qubits) >= 3:
            control_left = op.qubits[0]
            targets_left = op.qubits[1:]

            ##########################################################################################
            next_op_index = circuit.next_moment_operating_on(op.qubits[0], start_moment_index=index + 1)
            ##########################################################################################

            if next_op_index != index + 1:
                return None

            if next_op_index is not None:
                cnot = circuit.operation_at(op.qubits[0], next_op_index)
                if len(cnot.qubits) >= 3:

                    control_right = cnot.qubits[0]
                    targets_right = cnot.qubits[1:]

                    if control_left == control_right:
                        targets = [control_left] + targets_left + targets_right
                        gate = cirq.ParallelGate(cirq.X, len(targets[1:]))
                        c_op = gate.controlled().on(*targets)

                        new_op = [c_op]
                        print('I found Parallels to stick at ', index, next_op_index)

                        return cirq.PointOptimizationSummary(
                            clear_span=next_op_index - index + 1,
                            clear_qubits=targets,
                            new_operations=[new_op]
                        )
        return None