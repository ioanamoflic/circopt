import cirq
from optimization.optimize_circuits import CircuitIdentity
import global_stuff as g
import quantify.utils.misc_utils as mu
from optimization.parallel_point_optimizer import ParallelPointOptimizer


class ReverseCNOT(ParallelPointOptimizer):

    def __init__(self, only_count=False):
        super().__init__()

        self.only_count = only_count
        self.count = 0
        self.moment_index_qubit = []

    def optimization_at(self, circuit, index, op):

        if index != g.random_moment and not self.only_count:
            return None

        if mu.my_isinstance(op, cirq.CNOT):
            control = op.qubits[0]
            target = op.qubits[1]

            new_op = [cirq.H.on(control), cirq.H.on(target), cirq.CNOT.on(target, control), cirq.H.on(control), cirq.H.on(target)]

            if self.only_count:
                self.count += 1
                self.moment_index_qubit.append((CircuitIdentity.REVERSED_CNOT, index, control))
                return None

            return cirq.PointOptimizationSummary(
                clear_span=1,
                clear_qubits=[control, target],
                new_operations=[new_op]
            )
        return None
