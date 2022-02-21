import json
import pickle

from numpy import ndarray

from typing import List, Tuple, Union, Dict, Any

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
from optimization.optimize_circuits import CircuitIdentity
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

    initial_gate_count: int = 0
    for moment in initial_circuit:
        initial_gate_count += len(moment)

    final_gate_count: int = 0
    for moment in final_circuit:
        final_gate_count += len(moment)

    values = [initial_length, final_length, initial_gate_count, final_gate_count]
    names = ['Initial length', 'Final length', 'Initial gate count', 'Final gate count']
    fig, ax = plt.subplots()

    ax.bar(names, values)
    fig.savefig('evaluation.png', dpi=300)
    plt.show()


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

#  -------------------------------------------------- TEST UTILS --------------------------------------------------


def merge(q_table_1: np.ndarray, q_table_2: np.ndarray, state_map_1: dict, state_map_2: dict, action_map_1: dict,
          action_map_2: dict) -> Tuple[Union[ndarray, ndarray], Dict[Any, int], Dict[Any, int]]:
    merged_actions = set(list(action_map_1.keys()) + list(action_map_2.keys()))
    merged_states = set(list(state_map_1.keys()) + list(state_map_2.keys()))

    merged_qt: np.ndarray = np.zeros((len(merged_states), len(merged_actions)))
    merged_action_map = dict()
    merged_state_map = dict()

    for action in merged_actions:
        merged_action_map[action] = len(merged_action_map)

    for state in merged_states:
        merged_state_map[state] = len(merged_state_map)

    for state in merged_states:
        if state in state_map_1.keys() and state in state_map_2.keys():
            q_table_1_line_index = state_map_1[state]
            q_table_2_line_index = state_map_2[state]

            for action in merged_actions:
                if action in action_map_1.keys() and action in action_map_2.keys():
                    q_table_1_col_index = action_map_1[action]
                    q_table_2_col_index = action_map_2[action]
                    merged_qt[merged_state_map[state], merged_action_map[action]] = (q_table_1[
                                                                       q_table_1_line_index, q_table_1_col_index] +
                                                                   q_table_2[
                                                                       q_table_2_line_index, q_table_2_col_index]) / 2.0

                elif action in action_map_1.keys() and action not in action_map_2.keys():
                    q_table_1_col_index = action_map_1[action]
                    merged_qt[merged_state_map[state], merged_action_map[action]] = q_table_1[q_table_1_line_index, q_table_1_col_index]

                elif action in action_map_2.keys() and action not in action_map_1.keys():
                    q_table_2_col_index = action_map_2[action]
                    merged_qt[merged_state_map[state], merged_action_map[action]] = q_table_2[q_table_2_line_index, q_table_2_col_index]

        elif state in state_map_1.keys() and state not in state_map_2.keys():
            q_table_1_line_index = state_map_1[state]

            for action in merged_actions:
                if action in action_map_1.keys():
                    q_table_1_col_index = action_map_1[action]
                    merged_qt[merged_state_map[state], merged_action_map[action]] = q_table_1[q_table_1_line_index, q_table_1_col_index]

        elif state in state_map_2.keys() and state not in state_map_1.keys():
            q_table_2_line_index = state_map_2[state]

            for action in merged_actions:
                if action in action_map_2.keys():
                    q_table_2_col_index = action_map_2[action]
                    merged_qt[merged_state_map[state], merged_action_map[action]] = q_table_2[q_table_2_line_index, q_table_2_col_index]

    return merged_qt, merged_state_map, merged_action_map


def read_and_merge_all(q_tables: List[str], state_maps: List[str], action_maps: List[str]):
    assert len(q_tables) == len(state_maps) == len(action_maps)

    final_q_table, final_state_map, final_action_map = read_train_data(q_tables[0], action_maps[0], state_maps[0])

    for i in range(1, len(q_tables)):
        current_q_table, current_state_map, current_action_map = read_train_data(q_tables[i], action_maps[i],
                                                                                 state_maps[i])

        final_q_table, final_state_map, final_action_map = merge(final_q_table, current_q_table, final_state_map,
                                                                 current_state_map, final_action_map,
                                                                 current_action_map)

    return final_q_table, final_state_map, final_action_map


def read_train_data():
    q_table = np.load('train_data/QTable.npy')
    states = np.load('train_data/states.npy')
    actions = np.load('train_data/actions.npy')
    return q_table, states, actions


def write_train_data(q_table):
    np.save(f'train_data/QTable.npy', q_table)



