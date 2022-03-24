import numpy as np
from RL.circuit_env_identities import CircuitEnvIdent
import circopt_utils
import csv


class QAgentSingleCore:
    """
            Class used to train Q-learning agent on circuit optimization.
       """

    def __init__(self, env, n_ep=20000, max_iter=100, exploration_proba=1, expl_decay=0.001, min_expl_proba=0.01,
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
        self.Q_table = np.zeros((2, 2))
        self.state_map = dict()
        self.action_map = dict()
        circopt_utils.write_train_data(self.Q_table, self.state_map, self.action_map)

        # self.Q_table, self.state_map, self.action_map = circopt_utils.read_train_data()

    def train(self) -> None:
        """
        Performs off-policy control learning based on Bellman's equation.
        Q_table contains agent experience formed during training.
        :return: None
        """

        for e in range(self.n_episodes):
            current_state: str = self.env.reset()
            if current_state not in self.state_map.keys():
                self.state_map[current_state] = len(self.state_map)

            current_len: int = 0
            total_episode_reward: float = 0.0
            # print('Episode', e)

            for i in range(self.max_iter_episode):
                if np.random.uniform(0, 1) <= self.exploration_proba:
                    action = 'random'
                else:
                    action = np.argmax(self.Q_table[self.state_map.get(current_state), :])
                    for key, value in self.action_map.items():
                        if value == action:
                            action = key
                            break

                next_state, reward, done, info = self.env.step(action)

                if info["action"] not in self.action_map.keys():
                    self.action_map[info["action"]] = len(self.action_map)
                    self.Q_table = np.hstack((self.Q_table, np.zeros(self.Q_table.shape[0])[:, None]))

                if info["state"] not in self.state_map.keys():
                    self.state_map[info["state"]] = len(self.state_map)
                    self.Q_table = np.vstack((self.Q_table, np.zeros(self.Q_table.shape[1])))

                current_len = info["current_len"]
                next_state_index = self.state_map[info['state']]
                state_index = self.state_map[current_state]
                action_index = self.action_map[info['action']]

                self.Q_table[state_index, action_index] = (1 - self.learning_rate) * self.Q_table[
                    state_index, action_index] + self.learning_rate * (
                                                              reward + self.discount_factor * np.max(
                                                          self.Q_table[next_state_index, :]))

                total_episode_reward += reward
                current_state = next_state

                if done:
                    break

            self.exploration_proba = max(self.min_exploration_proba, np.exp(-self.exploration_decreasing_decay * e))
            self.rewards_per_episode.append(total_episode_reward)
            self.final_len_per_episode.append(current_len)

            circopt_utils.write_train_data(self.Q_table, self.state_map, self.action_map)

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

        header = ['nr_qbits', 'episode', 'reward', 'final_length', 'gamma', 'learning_rate', 'expl_decay']
        csv_lines = []

        with open(filename, 'w', encoding='UTF8', newline='\n') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for e in range(len(all_episodes)):
                csv_lines.append([bvz_bits, e, self.rewards_per_episode[e], self.final_len_per_episode[e],
                                  self.discount_factor, self.learning_rate, self.exploration_decreasing_decay]),
            writer.writerows(csv_lines)

        print("Mean reward per episode")
        for i in range(ep // 10):
            mean_rewards = np.append(mean_rewards, np.mean(self.rewards_per_episode[i * 10:(i + 1) * 10]))
            mean_lengths = np.append(mean_lengths, np.mean(self.final_len_per_episode[i * 10:(i + 1) * 10]))
            print((i + 1) * 10, ": Mean episode reward: ", np.mean(self.rewards_per_episode[i * 10:(i + 1) * 10]))
        print("\n\n")

        circopt_utils.plot(episodes, mean_rewards, "Episodes", "Mean rewards, lr=" + str(self.learning_rate) +
                                  " gamma=" + str(self.discount_factor), filename=filename)
        circopt_utils.plot(episodes, mean_lengths, "Episodes",
                               "Mean circuit length, lr=" + str(self.learning_rate) +
                               " gamma=" + str(self.discount_factor), filename=filename)
