from typing import Tuple

import numpy as np
from RL.circuit_env_identities import CircuitEnvIdent
import circopt_utils
import csv
import json
import global_stuff as g


class QAgent:
    """
            Class used to train Q-learning agent on circuit optimization.
       """

    def __init__(self, env, n_ep=20000, max_iter=100, exploration_proba=1, expl_decay=0.002, min_expl_proba=0.01,
                 gamma=0.99, lr=0.1):
        self.env: CircuitEnvIdent = env
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
        self.Q_table: np.ndarray = np.zeros(shape=(2, 2))

    def train(self, run_identifier: str) -> None:
        """
        Performs off-policy control learning based on Bellman's equation.
        Q_table contains agent experience formed during training.
        :return: None
        """

        for e in range(self.n_episodes):
            current_state: int = self.env.reset()
            current_len: int = 0
            current_gate_count: int = 0
            total_episode_reward: float = 0.0
            print('Episode', e)

            for i in range(self.max_iter_episode):
                if np.random.uniform(0, 1) <= self.exploration_proba:
                    action = 'random'
                else:
                    action = np.argmax(self.Q_table[current_state, :])

                next_state, reward, done, extra = self.env.step(action)

                # #Alexandru
                # import quantify.utils.misc_utils as mu
                # print(mu.my_isinstance.cache_info())
                # print(len(mu.my_cache), mu.my_cache_hits)

                current_len = extra["current_len"]
                action = extra['action']
                current_gate_count = extra['current_gate_count']

                self.Q_table[current_state, action] = (1 - self.learning_rate) * self.Q_table[
                    current_state, action] + self.learning_rate * (
                                                              reward + self.discount_factor * np.max(
                                                          self.Q_table[next_state, :]))

                total_episode_reward += reward
                current_state = next_state

                if self.Q_table.shape[0] <= len(g.state_map_identity.keys()):
                    self.Q_table = np.vstack((self.Q_table, np.zeros(self.Q_table.shape[1])))

                if self.Q_table.shape[1] <= len(g.action_map.keys()):
                    self.Q_table = np.hstack((self.Q_table, np.zeros(self.Q_table.shape[0])[:, None]))

                if done:
                    break

            self.exploration_proba = max(self.min_exploration_proba, np.exp(-self.exploration_decreasing_decay * e))
            self.rewards_per_episode.append(total_episode_reward)
            self.final_gc_per_episode.append(current_gate_count)
            self.final_len_per_episode.append(current_len)

            # save QTable state
            np.save(f'{run_identifier}_QTable.npy', self.Q_table)
            with open(f'{run_identifier}_State_Map.txt', 'w') as f1:
                json.dump(g.state_map_identity, f1)
            with open(f'{run_identifier}_Action_Map.txt', 'w') as f2:
                json.dump(str(g.action_map), f2)

    def show_evolution(self, filename: str = '3bits.csv', bvz_bits: int = 3, ep: int = 8000) -> None:
        """
        Prints mean cumulative rewards per 10 episodes.
        Plots mean rewards and mean circuit length per 10 episodes.
        :return: None
        """
        episodes: np.ndarray = np.arange(1, ep + 1, 10)
        all_episodes: np.ndarray = np.arange(1, ep + 1)
        mean_rewards: np.ndarray = np.array([])
        mean_lengths: np.ndarray = np.array([])
        mean_gate_count: np.ndarray = np.array([])

        header = ['nr_qbits', 'episode', 'reward', 'final_length', 'final_gate_count', 'gamma', 'learning_rate', 'expl_decay']
        csv_lines = []

        with open(filename, 'w', encoding='UTF8', newline='\n') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for e in range(len(all_episodes)):
                csv_lines.append([bvz_bits, e, self.rewards_per_episode[e], self.final_len_per_episode[e], self.final_gc_per_episode[e],
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
