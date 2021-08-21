# dqn
A PyTorch implementation of the DQN algorithm with the Double DQN (DDQN) improvement.

## Associated papers
```
@article{mnih_playing_2013,
	title = {Playing {Atari} with {Deep} {Reinforcement} {Learning}},
	url = {http://arxiv.org/abs/1312.5602},
	abstract = {We present the first deep learning model to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning. The model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards. We apply our method to seven Atari 2600 games from the Arcade Learning Environment, with no adjustment of the architecture or learning algorithm. We find that it outperforms all previous approaches on six of the games and surpasses a human expert on three of them.},
	urldate = {2021-08-21},
	journal = {arXiv:1312.5602 [cs]},
	author = {Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Graves, Alex and Antonoglou, Ioannis and Wierstra, Daan and Riedmiller, Martin},
	month = dec,
	year = {2013},
	note = {arXiv: 1312.5602},
	keywords = {Computer Science - Machine Learning},
	file = {arXiv Fulltext PDF:/Users/ryangr/Zotero/storage/ANV64XSM/Mnih et al. - 2013 - Playing Atari with Deep Reinforcement Learning.pdf:application/pdf;arXiv.org Snapshot:/Users/ryangr/Zotero/storage/WDPY5Y6P/1312.html:text/html},
}
```

```
@article{van_hasselt_deep_2015,
	title = {Deep {Reinforcement} {Learning} with {Double} {Q}-learning},
	url = {http://arxiv.org/abs/1509.06461},
	abstract = {The popular Q-learning algorithm is known to overestimate action values under certain conditions. It was not previously known whether, in practice, such overestimations are common, whether they harm performance, and whether they can generally be prevented. In this paper, we answer all these questions affirmatively. In particular, we first show that the recent DQN algorithm, which combines Q-learning with a deep neural network, suffers from substantial overestimations in some games in the Atari 2600 domain. We then show that the idea behind the Double Q-learning algorithm, which was introduced in a tabular setting, can be generalized to work with large-scale function approximation. We propose a specific adaptation to the DQN algorithm and show that the resulting algorithm not only reduces the observed overestimations, as hypothesized, but that this also leads to much better performance on several games.},
	urldate = {2021-08-21},
	journal = {arXiv:1509.06461 [cs]},
	author = {van Hasselt, Hado and Guez, Arthur and Silver, David},
	month = dec,
	year = {2015},
	note = {arXiv: 1509.06461},
	keywords = {Computer Science - Machine Learning},
	file = {arXiv Fulltext PDF:/Users/ryangr/Zotero/storage/Q42ZS9F7/van Hasselt et al. - 2015 - Deep Reinforcement Learning with Double Q-learning.pdf:application/pdf;arXiv.org Snapshot:/Users/ryangr/Zotero/storage/NKNFH2K8/1509.html:text/html},
}
```

```
@article{schaul_prioritized_2016,
	title = {Prioritized {Experience} {Replay}},
	url = {http://arxiv.org/abs/1511.05952},
	abstract = {Experience replay lets online reinforcement learning agents remember and reuse experiences from the past. In prior work, experience transitions were uniformly sampled from a replay memory. However, this approach simply replays transitions at the same frequency that they were originally experienced, regardless of their significance. In this paper we develop a framework for prioritizing experience, so as to replay important transitions more frequently, and therefore learn more efficiently. We use prioritized experience replay in Deep Q-Networks (DQN), a reinforcement learning algorithm that achieved human-level performance across many Atari games. DQN with prioritized experience replay achieves a new state-of-the-art, outperforming DQN with uniform replay on 41 out of 49 games.},
	urldate = {2021-08-21},
	journal = {arXiv:1511.05952 [cs]},
	author = {Schaul, Tom and Quan, John and Antonoglou, Ioannis and Silver, David},
	month = feb,
	year = {2016},
	note = {arXiv: 1511.05952},
	keywords = {Computer Science - Machine Learning},
	file = {arXiv Fulltext PDF:/Users/ryangr/Zotero/storage/QTFUYPGX/Schaul et al. - 2016 - Prioritized Experience Replay.pdf:application/pdf;arXiv.org Snapshot:/Users/ryangr/Zotero/storage/3X5533LB/1511.html:text/html},
}
```

```
@article{fortunato_noisy_2019,
	title = {Noisy {Networks} for {Exploration}},
	url = {http://arxiv.org/abs/1706.10295},
	abstract = {We introduce NoisyNet, a deep reinforcement learning agent with parametric noise added to its weights, and show that the induced stochasticity of the agent's policy can be used to aid efficient exploration. The parameters of the noise are learned with gradient descent along with the remaining network weights. NoisyNet is straightforward to implement and adds little computational overhead. We find that replacing the conventional exploration heuristics for A3C, DQN and dueling agents (entropy reward and \${\textbackslash}epsilon\$-greedy respectively) with NoisyNet yields substantially higher scores for a wide range of Atari games, in some cases advancing the agent from sub to super-human performance.},
	urldate = {2021-08-21},
	journal = {arXiv:1706.10295 [cs, stat]},
	author = {Fortunato, Meire and Azar, Mohammad Gheshlaghi and Piot, Bilal and Menick, Jacob and Osband, Ian and Graves, Alex and Mnih, Vlad and Munos, Remi and Hassabis, Demis and Pietquin, Olivier and Blundell, Charles and Legg, Shane},
	month = jul,
	year = {2019},
	note = {arXiv: 1706.10295},
	keywords = {Computer Science - Machine Learning, Statistics - Machine Learning},
	file = {arXiv Fulltext PDF:/Users/ryangr/Zotero/storage/RK6MJ9W6/Fortunato et al. - 2019 - Noisy Networks for Exploration.pdf:application/pdf;arXiv.org Snapshot:/Users/ryangr/Zotero/storage/NC99TQ5T/1706.html:text/html},
}
```

