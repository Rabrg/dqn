import random
import math
import gym


class EpsilonGreedyStrategy:
    def __init__(self):
        raise NotImplementedError

    def is_exploit(self):
        raise NotImplementedError


class FixedEpsilonGreedyStrategy(EpsilonGreedyStrategy):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def is_exploit(self):
        return random.uniform(0, 1) >= self.epsilon


class AnnealingEpsilonGreedyStrategy(EpsilonGreedyStrategy):
    def __init__(
        self,
        env: gym.Env,
        initial_exploration=1.0,
        final_exploration=0.05,
        exploration_decay=5000,
    ):
        self.env = env
        self.starting_epsilon = initial_exploration
        self.ending_epsilon = final_exploration
        self.epsilon_decay = exploration_decay

    def is_exploit(self):
        random_action_probability = self.ending_epsilon + (
            self.starting_epsilon - self.ending_epsilon
        ) * math.exp(-1.0 * self.env.get_total_steps() / self.epsilon_decay)
        return random.uniform(0, 1) >= random_action_probability
