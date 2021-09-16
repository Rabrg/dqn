import copy
import numpy as np
import torch
import fastprogress
import gym
from dqn.replay_memory import UniformReplayMemory
from dqn.epsilon_greedy_strategy import AnnealingEpsilonGreedyStrategy


class DQN:
    """
    A PyTorch implementation of DeepMind's DQN algorithm with the Double DQN (DDQN) improvement.
    """

    def __init__(
        self,
        env: gym.Env,
        model: torch.nn.Module,
        minibatch_size=128,
        target_update_frequency=100,
        use_ddqn_loss=True,
        discount_factor=0.99,
        update_frequency=1,
        learning_rate=1e-4,
        plot_update_frequency=10,
        max_episode_steps=108000,
    ):
        """
        Args:
            env (gym.Env): The environment that the DQN agent exists in. 
            model (torch.nn.Module): The neural network model used for Q-learning.
            minibatch_size (int, optional): The number of memories to train on during each SGD update. Defaults to 128.
            target_update_frequency (int, optional): The frequency at which the target network is updated (in episodes). Defaults to 100.
            use_ddqn_loss (bool, optional): Whether to use the loss function from the DDQN paper, or the original DQN loss function. Defaults to True.
            discount_factor (float, optional): The discount factor (gamma) that expresses the weighting of future rewards. Defaults to 0.99.
            update_frequency (int, optional): The frequence at which to train on recorded memories (in steps). Defaults to 1.
            lr (float, optional): The learning rate of the optimizer. Defaults to 1e-4.
            plot_update_frequency (int, optional): The frequency at which to update the fastprogress plot of rewards while training. Defaults to 10.
            max_episode_steps (int, optional): The maximum amount of steps to take before early exiting the episode. Defaults to 108000.
        """
        self.env = env
        self.model = model
        self.minibatch_size = minibatch_size
        self.target_update_frequency = target_update_frequency
        self.use_ddqn_loss = use_ddqn_loss
        self.discount_factor = discount_factor
        self.update_frequency = update_frequency
        self.plot_update_frequency = plot_update_frequency
        self.max_episode_steps = max_episode_steps

        self.replay_memory = UniformReplayMemory(memory_size=50000)
        self.epsilon_greedy_strategy = AnnealingEpsilonGreedyStrategy(env)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.done = True

        self._update_target_model_weights()

    def get_action(self, obs: np.ndarray, force_greedy=False) -> int:
        """
        If force_greedy is true this function will always return the expected optimal action. Otherwise, it may randomly sample from the action space according to the acting epsilon greedy strategy.

        Args:
            obs (np.ndarray): The most recent observation extracted from the environment,
            force_greedy (bool, optional): Whether a greedy action should be forced. Defaults to False.

        Returns:
            int: The discrete action returned from the current model / epsilon greedy strategy.
        """
        if force_greedy or self.epsilon_greedy_strategy.is_exploit():
            self.model.eval()
            with torch.no_grad():
                q_value_max = self.model(torch.Tensor(obs)).max(dim=0)
            self.model.train()
            return q_value_max[1].item()

        return self.env.action_space.sample()

    def _update_model_weights(self) -> float:
        batch = self.replay_memory.sample(self.minibatch_size)
        batch = {key: torch.stack([b[key] for b in batch]) for key in batch[0].keys()}

        q_value = (
            self.model(batch["obs"]).gather(1, batch["action"].unsqueeze(1)).squeeze(1)
        )
        target_next_q_values = self.target_model(batch["next_obs"])
        dones = batch["done"]

        if self.use_ddqn_loss:
            next_q_values = self.model(batch["next_obs"])
            next_q_value = target_next_q_values.gather(
                1, torch.max(next_q_values, 1)[1].unsqueeze(1)
            ).squeeze(1)
        else:
            next_q_value = target_next_q_values.max(dim=1)[0]

        target_q_value = batch["reward"] + self.discount_factor * next_q_value * (
            1 - dones
        )
        loss = torch.nn.functional.smooth_l1_loss(q_value, target_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()

    def _update_target_model_weights(self) -> None:
        self.target_model = copy.deepcopy(self.model).eval()

    def _step(self, obs: np.ndarray) -> tuple[np.ndarray, bool]:
        action = self.get_action(obs)
        next_obs, reward, done, _ = self.env.step(action)
        self.replay_memory.append(obs, action, reward, next_obs, done)
        return next_obs, done

    def learn(self, num_episodes=500) -> None:
        """
        Kicks off learning for the given number of training episodes. A fastprogress plot of historic rewards will be updated throughout training.

        Args:
            num_episodes (int, optional): The number of episodes to train for. Defaults to 500.
        """
        mb = fastprogress.master_bar(range(1, num_episodes + 1))
        for episode in mb:
            if self.done:
                self.obs = self.env.reset()

            for step in range(self.max_episode_steps):
                self.obs, self.done = self._step(self.obs)
                if (
                    len(self.replay_memory) >= self.minibatch_size
                    and step % self.update_frequency == 0
                ):
                    self._update_model_weights()
                if self.done:
                    break

            if episode % self.target_update_frequency == 0:
                self._update_target_model_weights()
            if episode % self.plot_update_frequency == 0:
                mb.update_graph(
                    [[range(1, episode + 1), self.env.get_episode_rewards()]],
                    [1 - 0.2, num_episodes + 0.2],
                )
        self.env.close()
