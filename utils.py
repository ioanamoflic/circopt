from typing import List

import cirq
import config as c
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from optimization.TopLeftT import TopLeftT
from optimization.TopRightT import TopRightT
from optimization.TopLeftHadamard import TopLeftHadamard
from optimization.OneHLeft2Right import OneHLeftTwoRight


def sort_tuple_list(tup):
    tup.sort(key=lambda x: x[1])
    return tup


def get_all_possible_identities(circuit):
    all_possibilities = []

    opt_circuit = OneHLeftTwoRight(only_count=True)
    opt_circuit.optimize_circuit(circuit)
    all_possibilities = all_possibilities + opt_circuit.moment_index

    opt_circuit = TopLeftT(only_count=True)
    opt_circuit.optimize_circuit(circuit)
    all_possibilities = all_possibilities + opt_circuit.moment_index

    opt_circuit = TopRightT(only_count=True)
    opt_circuit.optimize_circuit(circuit)
    all_possibilities = all_possibilities + opt_circuit.moment_index

    opt_circuit = TopLeftHadamard(only_count=True)
    opt_circuit.optimize_circuit(circuit)
    all_possibilities = all_possibilities + opt_circuit.moment_index

    return sort_tuple_list(all_possibilities)


def to_str(config: List[int]):
    current_config_as_string: str = "".join(
            [chr(ord('0') + config[i]) for i in range(len(config))])
    return current_config_as_string


def moment_has_toffoli(moment: cirq.Moment) -> bool:
    for op in moment.operations:
        if op.gate == cirq.TOFFOLI:
            return True
    return False


def get_data(bits: int) -> None:
    """
    Selects data used for Q-learning simulation and updates arrays from config file.
    :param bits: parameter user for choosing which adder circuit to train on
    :return: None
    """
    df: pd.DataFrame = pd.read_csv("data_letters.csv")
    c.all_configs = df.loc[df['nrbits'] == bits]['toffoli_desc_conf'].values
    c.input_depths = df.loc[df['nrbits'] == bits]['depth_input'].values
    c.output_depths = df.loc[df['nrbits'] == bits]['depth_output'].values
    c.ratios = c.input_depths / c.output_depths
    c.times = df.loc[df['nrbits'] == bits]['process_time'].values


def plot(x_axis: np.ndarray, y_axis: np.ndarray, xlabel: str, ylabel: str, decomp: str):
    fig1, ax1 = plt.subplots()
    lines, = ax1.plot(x_axis, y_axis)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    fig1.savefig(decomp + '.png', dpi=300)
    plt.close(fig1)




