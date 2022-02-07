import circopt_utils
import circopt_utils as utils
import numpy as np
import global_stuff as globals
from optimization.optimize_circuits import CircuitIdentity
from circuits.ioana_random import *


def optimize(test_circuit, Q_Table, state_map, action_map, steps=1):
    globals.action_map = action_map
    globals.state_map = state_map

    for step in range(steps):
        apply_on, current_state = circopt_utils.get_all_possible_identities(test_circuit)

        print(test_circuit)

        if current_state not in globals.state_map.keys():
            return test_circuit

        state_index = globals.state_map[current_state]
        best_action = np.argmax(Q_Table[state_index, :])
        action = utils.get_action_by_value(int(best_action))

        index = [index for index, value in enumerate(apply_on) if value[0].value == action[0] and value[2].name == action[1]]

        if len(index) == 0:
            return test_circuit

        if len(index) > 0:
            index = index[0]
            globals.random_moment = apply_on[index][1]

        if apply_on[index][0] == CircuitIdentity.REVERSED_CNOT:
            globals.working_optimizers["rerversecnot"].optimize_circuit(test_circuit)

        elif apply_on[index][0] == CircuitIdentity.ONE_HADAMARD_UP_LEFT:
            globals.working_optimizers["toplefth"].optimize_circuit(test_circuit)

        elif apply_on[index][0] == CircuitIdentity.ONE_HADAMARD_LEFT_DOUBLE_RIGHT:
            globals.working_optimizers["onehleft"].optimize_circuit(test_circuit)

        elif apply_on[index][0] == CircuitIdentity.DOUBLE_HADAMARD_LEFT_RIGHT:
            globals.working_optimizers["hadamardsquare"].optimize_circuit(test_circuit)

        test_circuit = circopt_utils.exhaust_optimization(test_circuit)

    return test_circuit


def run():
    q_tables = ['train_data/CR_0_QTable.npy', 'train_data/CR_1_QTable.npy',
                'train_data/CR_2_QTable.npy', 'train_data/CR_3_QTable.npy', 'train_data/CR_4_QTable.npy']

    action_maps = ['train_data/CR_0_Action_Map.txt', 'train_data/CR_1_Action_Map.txt',
                   'train_data/CR_2_Action_Map.txt', 'train_data/CR_3_Action_Map.txt',
                   'train_data/CR_4_Action_Map.txt']

    state_maps = ['train_data/CR_0_State_Map.txt', 'train_data/CR_1_State_Map.txt',
                  'train_data/CR_2_State_Map.txt', 'train_data/CR_3_State_Map.txt',
                  'train_data/CR_4_State_Map.txt']

    # final_q_table, final_state_map, final_action_map = utils.read_and_merge_all(q_tables, state_maps, action_maps)

    final_q_table, final_state_map, final_action_map = utils.read_train_data('train_data/test_circ_QTable.npy',
                                                                             'train_data/test_circ_Action_Map.txt',
                                                                             'train_data/test_circ_State_Map.txt')

    # test_file = 'test.txt'
    # f = open(test_file, 'r')
    # json_string = f.read()
    # test_circuit = cirq.read_json(json_text=json_string)

    optimized_circuit = optimize(get_test_circuit(), final_q_table, final_state_map, final_action_map)
    utils.plot_optimization_result(get_test_circuit(), optimized_circuit)


if __name__ == '__main__':
    run()
