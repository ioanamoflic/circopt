import csv

import numpy as np
import circopt_utils
from RL.multi_env import SubprocVecEnv


class QAgent:
    """
            Class used to train Q-learning agent on circuit optimization.
       """

    def __init__(self, vec_env, n_ep=20000, max_iter=100, exploration_proba=1, expl_decay=0.001, min_expl_proba=0.01,
                 gamma=0.99, lr=0.1):
        self.env: SubprocVecEnv = vec_env
        self.n_episodes: int = n_ep
        self.max_iter_episode: int = max_iter
        self.exploration_proba: int = exploration_proba
        self.exploration_decreasing_decay: float = expl_decay
        self.min_exploration_proba: float = min_expl_proba
        self.discount_factor: float = gamma
        self.learning_rate: float = lr
        self.rewards_per_episode = list()
        self.final_len_per_episode = list()
        self.final_gc_per_episode = list()

        self.Q_table, self.state_map, self.action_map = circopt_utils.read_train_data()

    def get_action_list(self, actions):
        action_list = []
        for action in actions:
            for key, value in self.action_map.items():
                if value == action:
                    action_list.append(key)

        return action_list

    def train(self) -> None:
        """
        Performs off-policy control learning based on Bellman's equation.
        Q_table contains agent experience formed during training.
        :return: None
        """

        current_states = self.env.reset()
        for state in current_states:
            if state not in self.state_map.keys():
                self.state_map[state] = len(self.state_map)
                self.Q_table = np.vstack((self.Q_table, np.zeros(self.Q_table.shape[1])))

        for e in range(self.n_episodes):
            current_states = self.env.reset()
            mean_len = 0
            mean_gate_count = 0
            mean_reward = 0

            for _ in range(self.max_iter_episode):
                if np.random.uniform(0, 1) <= self.exploration_proba:
                    actions = ['random'] * self.env.no_of_envs
                else:
                    actions = [np.argmax(self.Q_table[self.state_map.get(current_state), :])
                               for current_state in current_states]
                    actions = self.get_action_list(actions)

                next_states, rewards, dones, infos = self.env.step(actions)

                actions = []
                mean_reward = np.mean(rewards)
                mean_gate_count = np.array([])
                mean_len = np.array([])

                for info in infos:
                    if info["action"] not in self.action_map.keys():
                        self.action_map[info["action"]] = len(self.action_map)
                        self.Q_table = np.hstack((self.Q_table, np.zeros(self.Q_table.shape[0])[:, None]))

                    if info["state"] not in self.state_map.keys():
                        self.state_map[info["state"]] = len(self.state_map)
                        self.Q_table = np.vstack((self.Q_table, np.zeros(self.Q_table.shape[1])))

                    actions.append(info['action'])
                    mean_gate_count = np.append(mean_gate_count, info["current_gate_count"])
                    mean_len = np.append(mean_len, info["current_len"])

                mean_len = np.mean(mean_len)
                mean_gate_count = np.mean(mean_gate_count)

                for i in range(self.env.no_of_envs):
                    state_index = self.state_map[current_states[i]]
                    next_state_index = self.state_map[next_states[i]]
                    action_index = self.action_map[actions[i]]

                    self.Q_table[state_index, action_index] = (1 - self.learning_rate) * self.Q_table[
                        state_index, action_index] + self.learning_rate * (
                                                                      rewards[i] + self.discount_factor * np.max(
                                                                  self.Q_table[next_state_index, :]))

                current_states = next_states

            self.exploration_proba = max(self.min_exploration_proba, np.exp(-self.exploration_decreasing_decay * e))
            self.rewards_per_episode.append(mean_reward)
            self.final_gc_per_episode.append(mean_gate_count)
            self.final_len_per_episode.append(mean_len)

            if e % 5 == 0:
                circopt_utils.write_train_data(self.Q_table, self.state_map, self.action_map)

    def show_evolution(self, filename: str = '3bits.csv', bvz_bits: int = 3, ep: int = 8000) -> None:
        """
        Prints mean cumulative rewards per 10 episodes.
        Plots mean rewards and mean circuit length per 10 episodes.
        :return: None
        """
        episodes = np.arange(1, ep + 1, 10)
        all_episodes = np.arange(1, ep + 1)
        mean_rewards = np.array([])
        mean_lengths = np.array([])
        mean_gate_count = np.array([])

        header = ['nr_qbits', 'episode', 'reward', 'final_length', 'final_gate_count', 'gamma', 'learning_rate',
                  'expl_decay']
        csv_lines = []

        with open(filename, 'w', encoding='UTF8', newline='\n') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for e in range(len(all_episodes)):
                csv_lines.append([bvz_bits, e, self.rewards_per_episode[e], self.final_len_per_episode[e],
                                  self.final_gc_per_episode[e],
                                  self.discount_factor, self.learning_rate, self.exploration_decreasing_decay]),
            writer.writerows(csv_lines)

        print("Mean reward per episode")
        for i in range(ep // 10):
            mean_rewards = np.append(mean_rewards, np.mean(self.rewards_per_episode[i * 10:(i + 1) * 10]))
            mean_lengths = np.append(mean_lengths, np.mean(self.final_len_per_episode[i * 10:(i + 1) * 10]))
            mean_gate_count = np.append(mean_gate_count, np.mean(self.final_gc_per_episode[i * 10:(i + 1) * 10]))
            print((i + 1) * 10, ": Mean episode reward: ", np.mean(self.rewards_per_episode[i * 10:(i + 1) * 10]))
        print("\n\n")

        circopt_utils.plot(episodes, mean_rewards, "Episodes", "Mean rewards, lr=" + str(self.learning_rate) +
                           " gamma=" + str(self.discount_factor), filename='rewards.png')
        circopt_utils.plot(episodes, mean_lengths, "Episodes",
                           "Mean circuit length, lr=" + str(self.learning_rate) +
                           " gamma=" + str(self.discount_factor), filename='circuit_length.png')
        circopt_utils.plot(episodes, mean_gate_count, "Episodes",
                           "Mean gate count, lr=" + str(self.learning_rate) +
                           " gamma=" + str(self.discount_factor), filename='gate_count.png')
