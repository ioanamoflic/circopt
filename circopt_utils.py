import json
import pickle
from ast import literal_eval

from numpy import ndarray

from typing import List, Tuple, Union, Dict, Any
import json
import cirq
import pandas as pd
import global_stuff as g
import numpy as np
import matplotlib.pyplot as plt
from routing.routing_utils import plot_results
import quantify.optimizers as cnc

from optimization.stick_CNOTs import StickCNOTs
from optimization.stick_multitarget import StickMultiTarget
from optimization.stick_multitarget_to_CNOT import StickMultiTargetToCNOT

import logging

cancel_cnots = cnc.CancelNghCNOTs()
drop_empty = cirq.optimizers.DropEmptyMoments()
stick_cnots = StickCNOTs()
cancel_hadamards = cnc.CancelNghHadamards()
stick_multitarget = StickMultiTarget()
stick_to_cnot = StickMultiTargetToCNOT()

#  -------------------------------------------------- TRAIN UTILS --------------------------------------------------

def to_str(config: List[int]) -> str:
    current_config_as_string: str = "".join(
        [chr(ord('0') + config[i]) for i in range(len(config))])
    return current_config_as_string


def get_unique_representation(circuit) -> str:
    n_circuit = cirq.Circuit(circuit, strategy=cirq.InsertStrategy.EARLIEST)
    str_repr = str(n_circuit)
    # TODO: Alexandru maybe sha-1?
    return str_repr


def moment_has_toffoli(moment: cirq.Moment) -> bool:
    for op in moment.operations:
        if op.gate == cirq.TOFFOLI:
            return True
    return False


def optimize(circuit):
    cancel_cnots.optimize_circuit(circuit)
    drop_empty.optimize_circuit(circuit)
    stick_cnots.optimize_circuit(circuit)
    cancel_hadamards.optimize_circuit(circuit)
    stick_multitarget.optimize_circuit(circuit)
    drop_empty.optimize_circuit(circuit)
    stick_to_cnot.optimize_circuit(circuit)
    drop_empty.optimize_circuit(circuit)

    return circuit


def exhaust_optimization(circuit):
    prev_circ_repr: str = ""
    curr_circ_repr: str = get_unique_representation(circuit)

    while prev_circ_repr != curr_circ_repr:
        prev_circ_repr = curr_circ_repr
        circuit = optimize(circuit)
        curr_circ_repr = get_unique_representation(circuit)

    return circuit

#  -------------------------------------------------- PLOT UTILS --------------------------------------------------


def get_data(bits: int) -> None:
    df: pd.DataFrame = pd.read_csv("routing_data/data_letters.csv")
    g.all_configs = df.loc[df['nrbits'] == bits]['toffoli_desc_conf'].values
    g.input_depths = df.loc[df['nrbits'] == bits]['depth_input'].values
    g.output_depths = df.loc[df['nrbits'] == bits]['depth_output'].values
    g.ratios = g.input_depths / g.output_depths
    g.times = df.loc[df['nrbits'] == bits]['process_time'].values


def plot(x_axis: np.ndarray, y_axis: np.ndarray, xlabel: str, ylabel: str, filename: str) -> None:
    fig1, ax1 = plt.subplots()
    plt.style.use('seaborn')
    lines, = ax1.plot(x_axis, y_axis)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    fig1.savefig(filename, dpi=300)
    plt.close(fig1)


def plot_dataset():
    df: pd.DataFrame = pd.read_csv("routing_data/data_letters.csv")
    # output = (data['depth_output'].values[1:]).astype(float)
    # input = (data['depth_input'].values[1:]).astype(float)

    pd.columns = ['nrbits', 'toffoli_desc_conf', 'nr_qubits_output', 'depth_input', 'depth_output', 'process_time']

    grouped_df = df.groupby(["nrbits"])
    grouped_outputs = grouped_df['depth_output'].apply(list)
    grouped_lists = grouped_outputs.reset_index()
    outputs_av = np.mean(grouped_lists['depth_output'].tolist(), axis=1)

    grouped_df = df.groupby(["nrbits"])
    grouped_outputs = grouped_df['depth_input'].apply(list)
    grouped_lists = grouped_outputs.reset_index()
    inputs_av = np.mean(grouped_lists['depth_input'].tolist(), axis=1)

    grouped_df = df.groupby(["nrbits"])
    grouped_outputs = grouped_df['process_time'].apply(list)
    grouped_lists = grouped_outputs.reset_index()
    times_av = np.mean(grouped_lists['process_time'].tolist(), axis=1)

    depth_ratios = outputs_av / inputs_av

    print(times_av)
    print(depth_ratios)
    plot_results(times_av, depth_ratios, no_qubits=11, offset=4)


def plot_optimization_result(initial_circuit: cirq.Circuit, final_circuit: cirq.Circuit) -> None:
    initial_length = len(cirq.Circuit(initial_circuit.all_operations(), strategy=cirq.InsertStrategy.EARLIEST))
    final_length = len(cirq.Circuit(final_circuit.all_operations(), strategy=cirq.InsertStrategy.EARLIEST))

    print("Initial length: ", initial_length)
    print("Final length: ", final_length)

    initial_gate_count: int = 0
    for moment in initial_circuit:
        initial_gate_count += len(moment)

    final_gate_count: int = 0
    for moment in final_circuit:
        final_gate_count += len(moment)

    print("Initial gate count: ", initial_gate_count)
    print("Final gate count: ", final_gate_count)

    values = [initial_length, final_length, initial_gate_count, final_gate_count]
    names = ['Initial length', 'Final length', 'Initial gate count', 'Final gate count']
    fig, ax = plt.subplots()

    ax.bar(names, values)
    fig.savefig('evaluation.png', dpi=300)
    #plt.show()


def plot_reward_function():
    f = open('steps.txt', 'r')
    content = f.read()
    df = pd.read_json(content, lines=True)
    x = df['current_len']
    y = df['current_weight_av']
    z = df['reward']
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.set_xlabel('Length')
    ax.set_ylabel('Gate Weighted Average')
    ax.set_zlabel('Reward')

    ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5)

    ax.set_title('Reward function behaviour')
    fig.savefig('reward3D.png', dpi=300)
    #plt.show()


def plot_qt_size(x, y, z, colors, s:str):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    print(x)
    print(y)
    print(z)
    print(colors)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Partition size')
    ax.set_zlabel('Exp. Rates')

    ax.scatter(x, y, z, c=np.array(colors)/255, cmap='viridis', linewidth=0.5)

    ax.set_title('QTable size')
    if s == 'a':
        fig.savefig('qt3D_actions.png', dpi=300)
    else:
        fig.savefig('qt3D_states.png', dpi=300)

   # plt.show()


def read_train_data():
    q_table = np.load('train_data/QTable_2.npy')
    file1 = open('train_data/states_2.txt', 'r')
    file2 = open('train_data/actions_2.txt', 'r')
    state_map = json.load(file1)
    action_map_json = json.load(file2)
    action_map = {literal_eval(k): v for k, v in action_map_json.items()}

    return q_table, state_map, action_map


def write_train_data(q_table, state_map, action_map):
    np.save('train_data/QTable_2.npy', q_table)
    with open('train_data/states_2.txt', 'w') as f1:
        json.dump(state_map, f1)
    with open('train_data/actions_2.txt', 'w') as f2:
        json.dump({str((k[0], k[1], k[2], k[3])): v for k, v in action_map.items()}, f2)








