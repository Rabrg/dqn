from collections import deque
import random
import torch


class ReplayMemory:
    def __init__(self, memory_size):
        pass

    def append(self, obs, action, reward, next_obs, done):
        pass

    def sample(self, k):
        pass

    def __len__(self):
        pass


class UniformReplayMemory(ReplayMemory):
    def __init__(self, memory_size):
        self.memories = deque([], memory_size)

    def append(self, obs, action, reward, next_obs, done):
        self.memories.append(
            {
                "obs": torch.tensor(obs, dtype=torch.float),
                "action": torch.tensor(action, dtype=torch.long),
                "reward": torch.tensor(reward, dtype=torch.float),
                "next_obs": torch.tensor(next_obs, dtype=torch.float),
                "done": torch.tensor(done, dtype=torch.long),
            }
        )

    def sample(self, k):
        return random.sample(self.memories, k)

    def __len__(self):
        return len(self.memories)
