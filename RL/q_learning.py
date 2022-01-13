import numpy as np
from RL.circuit_env_identities import CircuitEnvIdent
import circopt_utils
import csv


class QAgent:
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
        self.Q_table: np.ndarray = np.zeros(shape=(10000, 10000))

    def train(self) -> None:
        """
        Performs off-policy control learning based on Bellman's equation.
        Q_table contains agent experience formed during training.
        :return: None
        """

        for e in range(self.n_episodes):
            current_state: int = self.env.reset()
            current_len: int = 0
            total_episode_reward: float = 0.0
            print('Episode', e)

            for i in range(self.max_iter_episode):
                if not self.env.could_apply_on:
                    break

                if np.random.uniform(0, 1) < self.exploration_proba:
                    identity, qubit = self.env.get_random_identity()
                    action = circopt_utils.get_random_action(identity, qubit)
                else:
                    action = np.argmax(self.Q_table[current_state, :])

                next_state, reward, done, extra = self.env.step(action)
                current_len = extra["current_len"]

                self.Q_table[current_state, action] = (1 - self.learning_rate) * self.Q_table[
                    current_state, action] + self.learning_rate * (
                                                              reward + self.discount_factor * np.max(
                                                          self.Q_table[next_state, :]))

                total_episode_reward += reward
                current_state = next_state

                if done:
                    break

            self.exploration_proba = max(self.min_exploration_proba, np.exp(-self.exploration_decreasing_decay * e))
            self.rewards_per_episode.append(total_episode_reward)
            self.final_len_per_episode.append(current_len)

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

        circopt_utils.plot_reward(episodes, mean_rewards, "Episodes", "Mean rewards, lr=" + str(self.learning_rate) +
                                  " gamma=" + str(self.discount_factor))
        circopt_utils.plot_len(episodes, mean_lengths, "Episodes",
                               "Mean circuit length, lr=" + str(self.learning_rate) +
                               " gamma=" + str(self.discount_factor))
