import cirq
from optimization.optimize_circuits import CircuitIdentity


class StickCNOTs(cirq.PointOptimizer):
    def __init__(self, optimize_till: int = None):
        super().__init__()
        self.optimize_till = optimize_till

    def optimization_at(self, circuit, index, op):
        if self.optimize_till is not None and index >= self.optimize_till:
            return None

        if isinstance(op, cirq.GateOperation) and (op.gate == cirq.CNOT):

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
                        targets = [qubit, op.qubits[1], target]
                        gate = cirq.ParallelGate(cirq.X, len(targets[1:]))
                        c_op = gate.controlled().on(*targets)

                        new_op = [c_op]
                        print('I found CNOTs to stick at ', index, next_op_index)

                        return cirq.PointOptimizationSummary(
                            clear_span=next_op_index - index + 1,
                            clear_qubits=targets,
                            new_operations=[new_op]
                        )
        return None