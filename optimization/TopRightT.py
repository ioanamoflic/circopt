import cirq
from optimization.optimize_circuits import CircuitIdentity


class TopRightT(cirq.PointOptimizer):
    def __init__(self, where_to: int = 0, only_count=False):
        super().__init__()
        self.where_to = where_to
        self.only_count = only_count
        self.count = 0
        self.moment_index = []

    def optimization_at(self, circuit, index, op):

        if index != self.where_to and not self.only_count:
            return None

        if isinstance(op, cirq.GateOperation) and (op.gate == cirq.CNOT):
            control = op.qubits[0]
            target = op.qubits[1]

            next_op_index = circuit.next_moment_operating_on(op.qubits, start_moment_index=index + 1)

            if next_op_index is not None:

                t_gate = circuit.operation_at(control, next_op_index)

                if t_gate is not None and isinstance(t_gate, cirq.GateOperation) and (t_gate.gate == cirq.T):
                    qubit = t_gate.qubits[0]

                    new_op = [cirq.T.on(qubit), cirq.CNOT.on(control, target)]

                    print('I found TopRightT ', index)

                    if self.only_count:
                        self.count += 1
                        self.moment_index.append((CircuitIdentity.T_GATE_RIGHT, index))
                        return None

                    return cirq.PointOptimizationSummary(
                        clear_span=next_op_index - index + 1,
                        clear_qubits=[control, target],
                        new_operations=[new_op]
                    )
        return None
