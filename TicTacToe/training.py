import os
import sys

import numpy as np

from util import *
from ReplayBuffer import ReplayBuffer
from dqn_agent import *
from TitTacToe import *

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.expanduser('~/SupervisedML/repos'))
from supervised_ml.plot_util import *


def main():
    tictactoe = TicTacToe(3, 3)
    renderer = CLIRenderer(tictactoe)

    filedSize = tictactoe.size * tictactoe.size
    env = TicTacToeEnv(tictactoe, RandomOpponent(filedSize), renderer)

    qnet = nn.Sequential(
        nn.Linear(filedSize, 12),
        nn.Linear(12, 12),
        nn.Linear(12, filedSize),
    )
    optimizer = optim.Adam(qnet.parameters(), lr=3.e-4, weight_decay=1.e-4)
    buffer = ReplayBuffer(filedSize, 1)
    agent = DQN(qnet, optimizer, buffer, filedSize)

    # n_episodes = 100_000
    n_episodes = 200
    losses = []
    rews = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_losses = []
        ep_rews = []
        while not done:
            act = agent.act(obs)
            next_obs, rew, done, _, _ = env.step(act)
            agent.store(obs, act, rew, next_obs, done)
            loss_val = agent.train()
            if loss_val is not None:
                ep_losses.append(loss_val)
            ep_rews.append(rew)
            obs = next_obs
        if ep_losses:
            losses.append(np.mean(ep_losses))
        rews.append(np.sum(ep_rews))

    basic_plot(np.arange(len(losses)) + 1, np.array(losses),
               xlabel='Episode', ylabel='loss',
               plotFilePath='results/basic_dqn/losses.png',
               save=True,)
    basic_plot(np.arange(len(losses)) + 1, np.log(np.array(losses)),
               xlabel='Episode', ylabel='log(loss)',
               plotFilePath='results/basic_dqn/log_losses.png',
               save=True, )
    basic_plot(np.arange(n_episodes) + 1, np.array(rews),
               xlabel='Episode', ylabel='reward',
               plotFilePath='results/basic_dqn/reward.png',
               save=True, )


if __name__ == '__main__':
    main()
