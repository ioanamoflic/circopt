from typing import List

import cirq
import pandas as pd
import global_stuff as g
import numpy as np
import matplotlib.pyplot as plt
from optimization.TopLeftT import TopLeftT
from optimization.TopRightT import TopRightT
from optimization.TopLeftHadamard import TopLeftHadamard
from optimization.OneHLeft2Right import OneHLeftTwoRight
from optimization.ReverseCNOT import ReverseCNOT
from optimization.HadamardSquare import HadamardSquare
from optimization.StickCNOTs import StickCNOTs


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

    opt_circuit = ReverseCNOT(only_count=True)
    opt_circuit.optimize_circuit(circuit)
    all_possibilities = all_possibilities + opt_circuit.moment_index

    opt_circuit = HadamardSquare(only_count=True)
    opt_circuit.optimize_circuit(circuit)
    all_possibilities = all_possibilities + opt_circuit.moment_index

    # opt_circuit = StickCNOTs(only_count=True)
    # opt_circuit.optimize_circuit(circuit)
    # all_possibilities = all_possibilities + opt_circuit.moment_index

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
    g.all_configs = df.loc[df['nrbits'] == bits]['toffoli_desc_conf'].values
    g.input_depths = df.loc[df['nrbits'] == bits]['depth_input'].values
    g.output_depths = df.loc[df['nrbits'] == bits]['depth_output'].values
    g.ratios = g.input_depths / g.output_depths
    g.times = df.loc[df['nrbits'] == bits]['process_time'].values


def get_unique_representation(circuit):
    n_circuit = cirq.Circuit(circuit, strategy=cirq.InsertStrategy.EARLIEST)
    str_repr = str(n_circuit)
    # TODO: Alexandru maybe sha-1?
    return str_repr


def plot(x_axis: np.ndarray, y_axis: np.ndarray, xlabel: str, ylabel: str, decomp: str):
    fig1, ax1 = plt.subplots()
    lines, = ax1.plot(x_axis, y_axis)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    # fig1.savefig(decomp + '.png', dpi=300)
    # TODO: Alexandru
    fig1.savefig('rewards.png', dpi=300)

    plt.close(fig1)
