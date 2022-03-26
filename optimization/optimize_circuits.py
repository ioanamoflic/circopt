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
    DOUBLE_HADAMARD_LEFT = 1
    DOUBLE_HADAMARD_RIGHT = 2
    DOUBLE_HADAMARD_UP = 3
    DOUBLE_HADAMARD_DOWN = 4
    DOUBLE_HADAMARD_LEFT_RIGHT = 5
    REVERSED_CNOT = 6
    ONE_HADAMARD_UP_LEFT = 7
    ONE_HADAMARD_LEFT_DOUBLE_RIGHT = 8
    T_GATE_LEFT = 9
    T_GATE_RIGHT = 10
    STICK_CNOTS = 11
    CANCEL_HADAMARDS = 12
    CANCEL_CNOTS = 13
    STICK_MULTITARGET = 14
    STICK_MULTITARGET_TO_CNOT = 15
