import matplotlib.pyplot as plt
import numpy as np


def plot_results(duration: np.ndarray, depths: np.ndarray, trials: int = 1, no_qubits: int = 3, offset: int = 3):
    """
    Plots results of routing a number of circuits in one or more trials.
    :param duration: np.ndarray containing routing process time of each circuit
    :param depths: np.ndarray containing depth ratio of each circuit
    :param trials: number of trials
    :param no_qubits: maximum number of qubits a circuit can have (+ offset)
    :param offset: starting point for number of qubits
    :return:
    """

    qubit_range = np.arange(no_qubits) + offset
    fig, axs = plt.subplots(2)
    fig.suptitle('Routing results')
    axs[0].set_xlabel('No. qubits')
    axs[0].set_ylabel('Av. depth ratio')
    axs[0].plot(qubit_range, depths / trials)

    axs[1].set_xlabel('No. qubits')
    axs[1].set_ylabel('Av. time (seconds)')
    axs[1].plot(qubit_range, duration / trials)
    plt.show()

