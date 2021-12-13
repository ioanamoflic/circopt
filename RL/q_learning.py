import numpy as np
import global_stuff as g
from RL.circuit_env_identities import CircuitEnvIdent
import circopt_utils
import random


class QAgent:
    """
            Class used to train Q-learning agent on circuit planning.
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
        self.Q_table: np.ndarray = np.zeros(shape=(10000, 1000))

    def train(self) -> None:
        """
        Performs off-policy control learning based on Bellman's equation. (greedy approach)
        Q_table contains agent experience formed during training.
        :return: None
        """

        for e in range(self.n_episodes):
            current_state: int = self.env.reset()
            current_len = 0
            total_episode_reward: float = 0.0
            print('Episode', e)

            for i in range(self.max_iter_episode):
                if np.random.uniform(0, 1) < self.exploration_proba:
                    action: int = circopt_utils.get_random_action(current_state, self.env.get_moment_to_apply_on())
                else:
                    action: int = np.argmax(self.Q_table[current_state, :])

                next_state, reward, done, extra = self.env.step(action)
                current_len = extra["current_len"]

                if done:
                    break

                self.Q_table[current_state, action] = (1 - self.learning_rate) * self.Q_table[
                    current_state, action] + self.learning_rate * (
                                                              reward + self.discount_factor * np.max(
                                                          self.Q_table[next_state, :]))

                total_episode_reward += reward
                current_state = next_state

            self.exploration_proba = max(self.min_exploration_proba, np.exp(-self.exploration_decreasing_decay * e))
            self.rewards_per_episode.append(total_episode_reward)
            self.final_len_per_episode.append(current_len)

    def show_evolution(self, conf: str) -> None:
        """
        Prints mean cumulative rewards per 10 episodes.
        :return: None
        """
        episodes = np.arange(1, 8001, 10)
        mean_rewards = np.array([])
        mean_lengths = np.array([])

        print("Mean reward per episode")
        for i in range(800):
            mean_rewards = np.append(mean_rewards, np.mean(self.rewards_per_episode[i * 10:(i + 1) * 10]))
            mean_lengths = np.append(mean_lengths, np.mean(self.final_len_per_episode[i * 10:(i + 1) * 10]))
            print((i + 1) * 10, ": Mean episode reward: ", np.mean(self.rewards_per_episode[i * 10:(i + 1) * 10]))
        print("\n\n")

        circopt_utils.plot_reward(episodes, mean_rewards, "Episodes", "Mean rewards, lr=0.2, gamma=0.98", conf)
        circopt_utils.plot_len(episodes, mean_lengths, "Episodes", "Mean circuit length, lr=0.2, gamma=0.98", conf)