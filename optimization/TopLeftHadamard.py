import cirq


class TopLeftHadamard(cirq.PointOptimizer):
    def __init__(self, where_to: int = 0, only_count=False):
        super().__init__()
        self.where_to = where_to
        self.only_count = only_count
        self.moment_index = []

    def optimization_at(self, circuit, index, op):
        if index != self.where_to and not self.only_count:
            return None

        if isinstance(op, cirq.GateOperation) and (op.gate == cirq.H):

            next_op_index = circuit.next_moment_operating_on(op.qubits, start_moment_index=index + 1)
            qubit = op.qubits[0]

            if next_op_index != index + 1:
                return None

            if next_op_index is not None:

                cnot = circuit.operation_at(qubit, next_op_index)

                if isinstance(cnot, cirq.GateOperation) and (cnot.gate == cirq.CNOT):
                    target = cnot.qubits[0]
                    control = cnot.qubits[1]

                    if qubit == control:

                        new_op = [cirq.H.on(control), cirq.CNOT.on(control, target), cirq.H.on(control), cirq.H.on(target)]

                        print('I found TopLeftHadamard ', index)

                        if self.only_count:
                            self.moment_index.append((1, index))
                            return None

                        return cirq.PointOptimizationSummary(
                            clear_span=next_op_index - index + 1,
                            clear_qubits=[control, target],
                            new_operations=[new_op]
                        )
        return None
