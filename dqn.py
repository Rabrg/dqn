import copy
import numpy as np
import torch
import fastprogress
import math


class DQN:
    def __init__(
        self,
        env,
        model,
        replay_memory=UniformReplayMemory(memory_size=50000),
        batch_size=128,
        target_model_update_delay=100,
        use_ddqn_loss=True,
        gamma=0.99,
        replay_period=1,
        lr=1e-4,
    ):
        self.env = env
        self.replay_memory = replay_memory
        self.epsilon_greedy_strategy = AnnealingEpsilonGreedyStrategy(env)
        self.model = model
        self.update_target_model_weights()
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.target_model_update_delay = target_model_update_delay
        self.use_ddqn_loss = use_ddqn_loss
        self.gamma = gamma
        self.replay_period = replay_period
        self.done = True

    def get_action(self, obs, force_greedy=False):
        if force_greedy or self.epsilon_greedy_strategy.is_exploit():
            self.model.eval()
            with torch.no_grad():
                q_value_max = self.model(torch.Tensor(obs)).max(dim=0)
            self.model.train()
            return q_value_max[1].item()
        return self.env.action_space.sample()

    def update_model_weights(self):
        batch = self.replay_memory.sample(self.batch_size)
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

        target_q_value = batch["reward"] + self.gamma * next_q_value * (1 - dones)

        loss = torch.nn.functional.smooth_l1_loss(q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def update_target_model_weights(self):
        self.target_model = copy.deepcopy(self.model).eval()

    def step(self, obs):
        action = self.get_action(obs)
        next_obs, reward, done, info = self.env.step(action)
        self.replay_memory.append(obs, action, reward, next_obs, done)
        return next_obs, done

    def plot_reward_update(self, epoch, epochs, mb, reward):
        x = range(1, epoch + 1)
        graphs = [[x, reward]]
        x_margin = 0.2
        y_margin = 0.05
        x_bounds = [1 - x_margin, epochs + x_margin]
        y_bounds = [np.min(reward) - y_margin, np.max(reward) + y_margin]

        mb.update_graph(graphs, x_bounds)

    def learn(self, n_episodes=500):
        mb = fastprogress.master_bar(range(1, n_episodes + 1))
        for episode in mb:
            if self.done:
                self.obs = self.env.reset()

            for step in range(108000):
                self.obs, self.done = self.step(self.obs)

                if (
                    len(self.replay_memory) >= self.batch_size
                    and step % self.replay_period == 0
                ):
                    self.update_model_weights()

                if self.done:
                    break
            if episode % self.target_model_update_delay == 0:
                self.update_target_model_weights()
            if episode % 10 == 0:
                self.plot_reward_update(
                    episode, n_episodes, mb, self.env.get_episode_rewards()
                )
            if episode >= 10 and np.mean(self.env.get_episode_rewards()[-10:]) >= 490:
                break
        self.env.close()
