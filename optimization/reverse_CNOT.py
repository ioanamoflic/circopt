import cirq
from optimization.optimize_circuits import CircuitIdentity
import quantify.utils.misc_utils as mu


class ReverseCNOT(cirq.PointOptimizer):
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
