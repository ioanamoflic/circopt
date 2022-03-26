import cirq
import quantify.utils.misc_utils as mu
from optimization.optimize_circuits import CircuitIdentity


class StickCNOTs(cirq.PointOptimizer):
    def __init__(self, moment=None, qubit=None, only_count=False, count_between=False):
        super().__init__()
        self.only_count = only_count
        self.count = 0
        self.moment_index_qubit = []
        self.moment = moment
        self.qubit = qubit
        self.start_moment = 0
        self.end_moment = 0
        self.count_between = count_between

    def optimization_at(self, circuit, index, op):
        if self.count_between and (index < self.start_moment or index > self.end_moment):
            return None

        if (index != self.moment or op.qubits[0] != self.qubit) and not self.only_count and not self.count_between:
            return None

        if hasattr(op, "allow"):
            return cirq.PointOptimizationSummary(
                clear_span=1,
                clear_qubits=op.qubits,
                new_operations=[]
            )

        if mu.my_isinstance(op, cirq.CNOT):
            next_op_index = circuit.next_moment_operating_on([op.qubits[0]], start_moment_index=index + 1)
            qubit = op.qubits[0]

            if next_op_index is not None:
                cnot = circuit.operation_at(qubit, next_op_index)
                if mu.my_isinstance(cnot, cirq.CNOT):
                    control = cnot.qubits[0]
                    target = cnot.qubits[1]
                    if qubit == control:
                        prev_op_index_target = circuit.prev_moment_operating_on([target],
                                                                                end_moment_index=next_op_index - 1)

                        if prev_op_index_target is not None and prev_op_index_target >= index:
                            return None

                        targets = [qubit, op.qubits[1], target]

                        if len(set(targets)) < len(targets):
                            return None

                        gate = cirq.ParallelGate(cirq.X, len(targets[1:]))
                        c_op = gate.controlled().on(*targets)

                        new_op = [c_op]

                        if self.count_between:
                            self.count += 1
                            return None

                        if self.only_count:
                            self.count += 1
                            self.moment_index_qubit.append(
                                (CircuitIdentity.STICK_CNOTS.value, index, op.qubits[0])
                            )
                            return None

                        # remove remaining op (cnot)
                        setattr(cnot, "allow", False)

                        return cirq.PointOptimizationSummary(
                            clear_span=1,
                            clear_qubits=targets,
                            new_operations=[new_op]
                        )
        return None
