import random
import math


class EpsilonGreedyStrategy:
    def __init__(self):
        pass

    def is_exploit(self):
        pass


class FixedEpsilonGreedyStrategy(EpsilonGreedyStrategy):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def is_exploit(self):
        return random.uniform(0, 1) >= self.epsilon


class AnnealingEpsilonGreedyStrategy(EpsilonGreedyStrategy):
    def __init__(
        self, env, starting_epsilon=1.0, ending_epsilon=0.05, epsilon_decay=5000
    ):
        self.env = env
        self.starting_epsilon = starting_epsilon
        self.ending_epsilon = ending_epsilon
        self.epsilon_decay = epsilon_decay

    def is_exploit(self):
        random_action_probability = self.ending_epsilon + (
            self.starting_epsilon - self.ending_epsilon
        ) * math.exp(-1.0 * self.env.get_total_steps() / self.epsilon_decay)
        return random.uniform(0, 1) >= random_action_probability
