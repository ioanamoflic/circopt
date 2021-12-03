import numpy as np
import gym
import config as c
from quantify.qramcircuits.toffoli_decomposition import ToffoliDecompType
import utils


class QAgent:
    """
            Class used to train Q-learning agent on circuit planning.
       """

    def __init__(self, env, n_ep=20000, max_iter=100, exploration_proba=1, expl_decay=0.001, min_expl_proba=0.01,
                 gamma=0.99, lr=0.1):
        self.env: gym.Env = env
        self.n_episodes: int = n_ep
        self.max_iter_episode: int = max_iter
        self.exploration_proba: int = exploration_proba
        self.exploration_decreasing_decay: float = expl_decay
        self.min_exploration_proba: float = min_expl_proba
        self.discount_factor: float = gamma
        self.learning_rate: float = lr
        self.rewards_per_episode = list()
        self.Q_table: np.ndarray = np.zeros(shape=(10000, env.action_space.n))

    def train(self) -> None:
        """
        Performs off-policy control learning based on Bellman's equation. (greedy approach)
        Q_table contains agent experience formed during training.
        :return: None
        """

        for e in range(self.n_episodes):
            current_state: int = self.env.reset()
            total_episode_reward: float = 0.0
            print('Episode', e)

            for i in range(self.max_iter_episode):
                if np.random.uniform(0, 1) < self.exploration_proba:
                    action: int = self.env.action_space.sample()
                else:
                    action: int = np.argmax(self.Q_table[current_state, :])

                next_state, reward, done, _ = self.env.step(action)

                if next_state == self.Q_table.shape[0]:
                    self.Q_table = np.vstack([self.Q_table, np.zeros(self.env.action_space.n)])

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

    def show_evolution(self, conf: str) -> None:
        """
        Prints mean cumulative reward per 1K episodes.
        :return: None
        """
        print(self.Q_table)
        episodes = np.arange(1, 11, 1)
        mean_rewards = np.array([])
        print("Mean reward per episode")
        for i in range(10):
            mean_rewards = np.append(mean_rewards, np.mean(self.rewards_per_episode[i:(i + 1)]))
            print((i + 1), ": mean espiode reward: ", np.mean(self.rewards_per_episode[i:(i + 1)]))
        print("\n\n")
        utils.plot(episodes, mean_rewards, "Episodes", "Mean rewards", conf)
