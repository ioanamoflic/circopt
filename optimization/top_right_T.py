import cirq
from optimization.optimize_circuits import CircuitIdentity
from optimization.parallel_point_optimizer import ParallelPointOptimizer

import global_stuff as g
import quantify.utils.misc_utils as mu


class TopRightT(ParallelPointOptimizer):
    def __init__(self, only_count=False):
        super().__init__(only_count=only_count)

    def optimization_at(self, circuit, index, op):

        if index != g.random_moment and not self.only_count:
            return None

        if mu.my_instance(op, cirq.CNOT):
            control = op.qubits[0]
            target = op.qubits[1]

            next_op_index = circuit.next_moment_operating_on(op.qubits, start_moment_index=index + 1)

            if next_op_index is not None:

                t_gate = circuit.operation_at(control, next_op_index)

                if t_gate is not None and mu.my_instance(t_gate, cirq.T):
                    qubit = t_gate.qubits[0]

                    new_op = [cirq.T.on(qubit), cirq.CNOT.on(control, target)]

                    if self.only_count:
                        self.increase_opt_counter(CircuitIdentity.T_GATE_RIGHT, index, -1)
                        return None

                    return cirq.PointOptimizationSummary(
                        clear_span=next_op_index - index + 1,
                        clear_qubits=[control, target],
                        new_operations=[new_op]
                    )
        return None
