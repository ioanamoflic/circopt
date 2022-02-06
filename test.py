import circopt_utils as utils


def run():
    q_tables = ['train_data/CR_0_QTable.npy', 'train_data/CR_1_QTable.npy',
                'train_data/CR_2_QTable.npy', 'train_data/CR_3_QTable.npy', 'train_data/CR_4_QTable.npy']

    action_maps = ['train_data/CR_0_Action_Map.txt', 'train_data/CR_1_Action_Map.txt',
                   'train_data/CR_2_Action_Map.txt', 'train_data/CR_3_Action_Map.txt',
                   'train_data/CR_4_Action_Map.txt']

    state_maps = ['train_data/CR_0_State_Map.txt', 'train_data/CR_1_State_Map.txt',
                  'train_data/CR_2_State_Map.txt', 'train_data/CR_3_State_Map.txt',
                  'train_data/CR_4_State_Map.txt']

    final_q_table, final_state_map, final_action_map = utils.read_and_merge_all(q_tables, state_maps, action_maps)

    print(final_q_table)
    print(final_state_map)
    print(final_action_map)


if __name__ == '__main__':
    run()
