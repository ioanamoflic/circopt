from typing import Optional, Tuple, List

import cirq
from enum import Enum, auto, EnumMeta

identity_pairs = {6: 7, 8: 9, 7: 6, 9: 8}


def get_each_identity(circuit_identity, qubit1, qubit2):
    moments = []
    if circuit_identity == CircuitIdentity.DOUBLE_HADAMARD_LEFT:
        moments += [
            cirq.Moment([cirq.H.on(qubit1), cirq.H.on(qubit2)]),
            cirq.Moment([cirq.CNOT.on(qubit1, qubit2)]),
        ]
        return moments

    if circuit_identity == CircuitIdentity.DOUBLE_HADAMARD_RIGHT:
        moments += [
            cirq.Moment([cirq.CNOT.on(qubit2, qubit1)]),
            cirq.Moment([cirq.H.on(qubit1), cirq.H.on(qubit2)])
        ]
        return moments

    if circuit_identity == CircuitIdentity.DOUBLE_HADAMARD_UP:
        moments += [
            cirq.Moment([cirq.H.on(qubit1)]),
            cirq.Moment([cirq.CNOT.on(qubit1, qubit2)]),
            cirq.Moment([cirq.H.on(qubit1)])
        ]
        return moments

    if circuit_identity == CircuitIdentity.DOUBLE_HADAMARD_DOWN:
        moments += [
            cirq.Moment([cirq.H.on(qubit2)]),
            cirq.Moment([cirq.CNOT.on(qubit2, qubit1)]),
            cirq.Moment([cirq.H.on(qubit2)]),
        ]
        return moments

    if circuit_identity == CircuitIdentity.DOUBLE_HADAMARD_LEFT_RIGHT:
        moments += [
            cirq.Moment([cirq.H.on(qubit1), cirq.H.on(qubit2)]),
            cirq.Moment([cirq.CNOT.on(qubit1, qubit2)]),
            cirq.Moment([cirq.H.on(qubit1), cirq.H.on(qubit2)]),
        ]
        return moments

    if circuit_identity == CircuitIdentity.ONE_HADAMARD_UP_LEFT:
        moments += [
            cirq.Moment([cirq.H.on(qubit1)]),
            cirq.Moment([cirq.CNOT.on(qubit1, qubit2)])
        ]
        return moments

    if circuit_identity == CircuitIdentity.ONE_HADAMARD_LEFT_DOUBLE_RIGHT:
        moments += [
            cirq.Moment([cirq.H.on(qubit2)]),
            cirq.Moment([cirq.CNOT.on(qubit2, qubit1)]),
            cirq.Moment([cirq.H.on(qubit1), cirq.H.on(qubit2)]),
        ]
        return moments

    if circuit_identity == CircuitIdentity.T_GATE_LEFT:
        moments += [
            cirq.Moment([cirq.T.on(qubit1)]),
            cirq.Moment([cirq.CNOT.on(qubit1, qubit2)])
        ]
        return moments

    if circuit_identity == CircuitIdentity.REVERSED_CNOT:
        return cirq.Moment([cirq.CNOT.on(qubit2, qubit1)])

    if circuit_identity == CircuitIdentity.T_GATE_RIGHT:
        moments += [
            cirq.Moment([cirq.CNOT.on(qubit1, qubit2)]),
            cirq.Moment([cirq.T.on(qubit1)])
        ]
    return moments


class CircuitIdentity(Enum):
    DOUBLE_HADAMARD_LEFT = auto()
    DOUBLE_HADAMARD_RIGHT = auto()
    DOUBLE_HADAMARD_UP = auto()
    DOUBLE_HADAMARD_DOWN = auto()
    DOUBLE_HADAMARD_LEFT_RIGHT = auto()
    REVERSED_CNOT = auto()
    ONE_HADAMARD_UP_LEFT = auto()
    ONE_HADAMARD_LEFT_DOUBLE_RIGHT = auto()
    T_GATE_LEFT = auto()
    T_GATE_RIGHT = auto()
    STICK_CNOTS = auto()
