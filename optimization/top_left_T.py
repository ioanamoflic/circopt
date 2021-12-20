import cirq
from optimization.optimize_circuits import CircuitIdentity


class TopLeftT(cirq.PointOptimizer):
    def __init__(self, where_to: int = 0, only_count=False):
        super().__init__()
        self.where_to: int = where_to
        self.only_count: bool = only_count
        self.count: int = 0
        self.moment_index = list()

    def optimization_at(self, circuit, index, op):
        if index != self.where_to and not self.only_count:
            return None
        if isinstance(op, cirq.GateOperation) and (op.gate == cirq.T):

            next_op_index = circuit.next_moment_operating_on(op.qubits, start_moment_index=index + 1)

            if next_op_index != index + 1:
                return None

            qubit = op.qubits[0]

            if next_op_index is not None:

                cnot = circuit.operation_at(qubit, next_op_index)

                if isinstance(cnot, cirq.GateOperation) and (cnot.gate == cirq.CNOT):
                    control = cnot.qubits[0]
                    target = cnot.qubits[1]

                    if qubit == control:
                        new_op = [cirq.CNOT.on(control, target), cirq.T.on(op.qubits[0])]

                        if self.only_count:
                            self.count += 1
                            self.moment_index.append((CircuitIdentity.T_GATE_LEFT, index))
                            return None

                        return cirq.PointOptimizationSummary(
                            clear_span=next_op_index - index + 1,
                            clear_qubits=[control, target],
                            new_operations=[new_op]
                        )
        return None
