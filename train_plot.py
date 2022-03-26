import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import circopt_utils


def plot_reward():
    filename = sys.argv[1]
    columns = ['episode', 'reward', 'max_degree', 'current_degree', 'max_len', 'current_len', 'min_w_av', 'current_w_av',
               'filename']
    df = pd.read_csv("logfile.log", sep=',', header=0, names=columns, index_col=False)
    df = df.loc[df['filename'] == filename]
    df = df.groupby(df['episode']).mean()

    circopt_utils.plot(np.arange(1, len(df.iloc[:, 0].values) + 1), df.iloc[:, 0].values,
                       'Episodes', 'Reward', f'train_plots/episodes_{filename}.png')


def plot_qt_size():
    df = pd.read_csv("logfile_QTable.log", sep=',', header=0)
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    z = df.iloc[:, 2].values
    states = df.iloc[:, 3].values
    actions = df.iloc[:, 4].values

    fig = plt.figure(figsize=(12, 6))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Partition size')
    ax.set_zlabel('No. steps')
    f1 = ax.scatter(x, y, z, c=np.array(states), cmap='viridis', linewidth=0.5)
    plt.colorbar(f1)
    ax.set_title('States')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Partition size')
    ax.set_zlabel('No. steps')
    f2 = ax.scatter(x, y, z, c=np.array(actions), cmap='viridis', linewidth=0.5)
    plt.colorbar(f2)
    ax.set_title('Actions')

    plt.tight_layout()

    plt.show()


def plot_training_for_circuit():
    filename = sys.argv[1]
    columns = ['e', 'reward', 'max_degree', 'current_degree', 'max_len', 'current_len', 'min_w_av', 'current_w_av', 'filename']
    df = pd.read_csv("logfile.log", sep=',', header=0, names=columns, index_col=False)
    df = df.loc[df['filename'] == filename]
    x = df['current_degree']
    y = df['current_len']
    z = df['reward']
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.set_xlabel('Circuit Degree')
    ax.set_ylabel('Circuit Length')
    ax.set_zlabel('Reward')

    ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5)

    ax.set_title('Reward function behaviour')
    fig.savefig(f'train_plots/reward3D_{filename}.png', dpi=300)
    #plt.show()


if __name__ == '__main__':
    plot_reward()
    plot_training_for_circuit()
