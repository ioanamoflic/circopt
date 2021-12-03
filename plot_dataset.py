from routing.routing_utils import plot_results
import pandas as pd
import numpy as np

df: pd.DataFrame = pd.read_csv("data_letters.csv")
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

