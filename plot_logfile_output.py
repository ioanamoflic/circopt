import sys
import pandas as pd
import matplotlib.pyplot as plt


def plot_training_for_circuit():
    filename = sys.argv[1]
    columns = ['reward', 'max_degree', 'current_degree', 'max_len', 'current_len', 'min_w_av', 'current_w_av', 'filename']
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
    fig.savefig('reward3D.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    plot_training_for_circuit()
