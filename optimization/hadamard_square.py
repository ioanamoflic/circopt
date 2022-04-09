from typing import List, Tuple

import cirq
from optimization.optimize_circuits import CircuitIdentity
from ..quantify.utils import misc_utils as mu


class HadamardSquare(cirq.PointOptimizer):
    def __init__(self, moment=None, qubits=None, only_count=False, count_between=False):
        super().__init__()
        self.only_count: bool = only_count
        self.count: int = 0
        self.moment_index_qubit: List[Tuple[int, int, List[cirq.Qid]]] = []
        self.moment: int = moment
        self.qubits: List[cirq.NamedQubit] = qubits
        self.start_moment: int = 0
        self.end_moment: int = 0
        self.count_between: bool = count_between

    def optimization_at(self, circuit, index, op):
        if self.count_between and (index < self.start_moment or index > self.end_moment):
            return None

        if hasattr(op, "delete_CNOT") and not self.only_count and not self.count_between:
            delattr(op, "delete_CNOT")
            return cirq.PointOptimizationSummary(
                clear_span=1,
                clear_qubits=op.qubits,
                new_operations=[cirq.CNOT.on(op.qubits[1], op.qubits[0])]
            )

        if hasattr(op, "delete_H") and not self.only_count and not self.count_between:
            delattr(op, "delete_H")
            return cirq.PointOptimizationSummary(
                clear_span=1,
                clear_qubits=op.qubits,
                new_operations=[]
            )

        if (index != self.moment or op.qubits[0] not in self.qubits) and not self.only_count and not self.count_between:
            return None

        # if index != self.moment and not self.only_count and not self.count_between:
        #     return None

        if mu.my_isinstance(op, cirq.H):

            cnot_moment = circuit.next_moment_operating_on(op.qubits, start_moment_index=index + 1)
            qubit = op.qubits[0]

            if cnot_moment is not None:
                cnot = circuit.operation_at(qubit, cnot_moment)
                if mu.my_isinstance(cnot, cirq.CNOT):
                    control = cnot.qubits[0]
                    target = cnot.qubits[1]

                    h_right_control_index = circuit.next_moment_operating_on([control],
                                                                             start_moment_index=cnot_moment + 1)
                    h_right_target_index = circuit.next_moment_operating_on([target],
                                                                            start_moment_index=cnot_moment + 1)

                    h_left_control_index = circuit.prev_moment_operating_on([control], end_moment_index=cnot_moment)
                    h_left_target_index = circuit.prev_moment_operating_on([target], end_moment_index=cnot_moment)

                    if h_right_target_index is not None and h_right_control_index is not None and \
                            h_left_control_index is not None and h_left_target_index is not None:

                        h_right_control = circuit.operation_at(control, h_right_control_index)
                        h_right_target = circuit.operation_at(target, h_right_target_index)
                        h_left_control = circuit.operation_at(control, h_left_control_index)
                        h_left_target = circuit.operation_at(target, h_left_target_index)

                        if mu.my_isinstance(h_right_control, cirq.H) and mu.my_isinstance(h_right_target, cirq.H) and \
                                mu.my_isinstance(h_left_control, cirq.H) and mu.my_isinstance(h_left_target, cirq.H):

                            setattr(h_right_control, "delete_H", True)
                            setattr(h_right_target, "delete_H", True)
                            setattr(h_left_control, "delete_H", True)
                            setattr(h_left_target, "delete_H", True)
                            setattr(cnot, "delete_CNOT", True)

                            if self.count_between:
                                self.count += 1
                                return None

                            if self.only_count:
                                self.count += 1
                                self.moment_index_qubit.append(
                                    (CircuitIdentity.DOUBLE_HADAMARD_LEFT_RIGHT.value, index, [control, target])
                                )
                                return None

                            return cirq.PointOptimizationSummary(
                                clear_span=1,
                                clear_qubits=[qubit],
                                new_operations=[]
                            )

        return None
