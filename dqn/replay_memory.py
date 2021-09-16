from collections import deque
import numpy as np
import random
import torch


# TODO: #1 Implement prioritized replay (Schaul et al., 2016)
class ReplayMemory:
    def __init__(self, memory_size: int):
        raise NotImplementedError

    def append(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        raise NotImplementedError

    def sample(self, k) -> dict:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


# TODO: #3 Tensorize the uniform replay memory 
class UniformReplayMemory(ReplayMemory):
    def __init__(self, memory_size: int):
        self.memories = deque([], memory_size)

    def append(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.memories.append(
            {
                "obs": torch.tensor(obs, dtype=torch.float),
                "action": torch.tensor(action, dtype=torch.long),
                "reward": torch.tensor(reward, dtype=torch.float),
                "next_obs": torch.tensor(next_obs, dtype=torch.float),
                "done": torch.tensor(done, dtype=torch.long),
            }
        )

    def sample(self, k) -> dict:
        return random.sample(self.memories, k)

    def __len__(self) -> int:
        return len(self.memories)
