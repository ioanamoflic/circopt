import cirq
from optimization.optimize_circuits import CircuitIdentity
import global_stuff as g


class ReverseCNOT(cirq.PointOptimizer):
    def __init__(self, where_to: int = 0, only_count=False):
        super().__init__()
        self.where_to = where_to
        self.only_count = only_count
        self.moment_index = []

    def optimization_at(self, circuit, index, op):
        if index != self.where_to and not self.only_count:
            return None

        if isinstance(op, cirq.GateOperation) and (op.gate == cirq.CNOT):
            control = op.qubits[0]
            target = op.qubits[1]

            new_op = [cirq.H.on(control), cirq.H.on(target), cirq.CNOT.on(target, control), cirq.H.on(control), cirq.H.on(target)]

            print('I found CNOT to reverse ', index)

            if self.only_count:
                self.moment_index.append((CircuitIdentity.REVERSED_CNOT, index))
                return None

            return cirq.PointOptimizationSummary(
                clear_span=1,
                clear_qubits=[control, target],
                new_operations=[new_op]
            )
        return None