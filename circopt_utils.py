from typing import List, Tuple, Union
import cirq
import pandas as pd
import global_stuff as g
import numpy as np
import matplotlib.pyplot as plt
from optimization.top_left_T import TopLeftT
from optimization.one_H_left_2_right import OneHLeftTwoRight
from optimization.reverse_CNOT import ReverseCNOT
from optimization.hadamard_square import HadamardSquare
from optimization.top_left_hadamard import TopLeftHadamard
from optimization.one_H_left_2_right import OneHLeftTwoRight
import random


def get_random_action(current_state, moment) -> int:
    tuple = (current_state, moment, random.randint(0, 1))
    if tuple not in g.action_map.keys():
        g.action_map[tuple] = len(g.action_map)
    return g.action_map.get(tuple)


def get_action_by_value(value: int) -> Union[Tuple[int, int, int], None]:
    for tuple, index in g.action_map.items():
        if index == value:
            return tuple
    return None


def sort_tuple_list(tup):
    tup.sort(key=lambda x: x[1])
    return tup


def get_all_possible_identities(circuit):
    all_possibilities = []

    # opt_circuit = OneHLeftTwoRight(only_count=True)
    # opt_circuit.optimize_circuit(circuit)
    # all_possibilities = all_possibilities + opt_circuit.moment_index
    #
    # opt_circuit = TopLeftT(only_count=True)
    # opt_circuit.optimize_circuit(circuit)
    # all_possibilities = all_possibilities + opt_circuit.moment_index

    opt_circuit = OneHLeftTwoRight(only_count=True)
    opt_circuit.optimize_circuit(circuit)
    all_possibilities = all_possibilities + opt_circuit.moment_index

    opt_circuit = TopLeftHadamard(only_count=True)
    opt_circuit.optimize_circuit(circuit)
    all_possibilities = all_possibilities + opt_circuit.moment_index

    opt_circuit = ReverseCNOT(only_count=True)
    opt_circuit.optimize_circuit(circuit)
    all_possibilities = all_possibilities + opt_circuit.moment_index

    opt_circuit = HadamardSquare(only_count=True)
    opt_circuit.optimize_circuit(circuit)
    all_possibilities = all_possibilities + opt_circuit.moment_index

    return sort_tuple_list(all_possibilities)


def to_str(config: List[int]) -> str:
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


def get_unique_representation(circuit) -> str:
    n_circuit = cirq.Circuit(circuit, strategy=cirq.InsertStrategy.EARLIEST)
    str_repr = str(n_circuit)
    # TODO: Alexandru maybe sha-1?
    return str_repr


def plot_reward(x_axis: np.ndarray, y_axis: np.ndarray, xlabel: str, ylabel: str) -> None:
    fig1, ax1 = plt.subplots()
    plt.style.use('seaborn')
    lines, = ax1.plot(x_axis, y_axis)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    fig1.savefig('rewards.png', dpi=300)

    plt.close(fig1)


def plot_len(x_axis: np.ndarray, y_axis: np.ndarray, xlabel: str, ylabel: str) -> None:
    fig1, ax1 = plt.subplots()
    plt.style.use('seaborn')
    lines, = ax1.plot(x_axis, y_axis)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    fig1.savefig('circuit_length.png', dpi=300)

    plt.close(fig1)
