# Test files for a part of Bernstein Vazirani circuits

import cirq

def make_oracle(input_qubits, output_qubit, secret):
    for qubit, bit in zip(input_qubits, secret):
        if bit:
            yield cirq.CNOT(qubit, output_qubit)

def bernstein_vazirani(nr_bits = 2, secret = "11"):
  # n - number of bits
  # secret - the bit string called 'a' in some of the algorithm descriptions
  # Returns a n+1 qubit circuit, where the (n+1)-th qubit is for phase kickback
  # The data type does not matter, but it would be easier to use the Cirq/Qiskit
  # circuit data types

  circuit = None

  input_qubits = [cirq.GridQubit(i, 0) for i in range(nr_bits)]
  output_qubit = cirq.GridQubit(nr_bits, 0)

  circuit = cirq.Circuit()

  circuit.append(
       [
          cirq.X(output_qubit),
          cirq.H(output_qubit),
          cirq.H.on_each(input_qubits),
      ]
  )

  oracle = make_oracle(input_qubits, output_qubit, secret)

  circuit.append(oracle)

  circuit.append([cirq.H.on_each(input_qubits), cirq.measure(*input_qubits, key='result')])

  return circuit