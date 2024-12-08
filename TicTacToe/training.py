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


def run_episode(env, agent, train=True):
    if train:
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
        if not ep_losses:
            return None, np.sum(ep_rews)
        return np.mean(ep_losses), np.sum(ep_rews)

    else:
        obs, _ = env.reset()
        done = False
        ep_rews = []
        while not done:
            act = agent.act(obs, train=False)
            next_obs, rew, done, _, _ = env.step(act)
            ep_rews.append(rew)
            obs = next_obs
        return None, np.sum(ep_rews)


def main():
    tictactoe = TicTacToe(3, 3)
    renderer = CLIRenderer(tictactoe)

    fieldSize = tictactoe.size * tictactoe.size
    env = TicTacToeEnv(tictactoe, RandomOpponent(fieldSize), renderer)

    qnet = nn.Sequential(
        nn.Linear(fieldSize, 12),
        nn.Linear(12, 12),
        nn.Linear(12, fieldSize),
    )
    optimizer = optim.Adam(qnet.parameters(), lr=3.e-4, weight_decay=1.e-4)
    buffer = ReplayBuffer(fieldSize, 1)
    agent = DQN(qnet, optimizer, buffer, fieldSize)

    n_episodes = 100_000
    # n_episodes = 200
    evalPeriod = 1000
    eval_episodes = 100
    max_rew = -np.inf
    losses = []
    rews = []
    for i in range(n_episodes):
        loss_mean, rew_sum = run_episode(env, agent)
        if loss_mean is not None:
            losses.append(loss_mean)
        rews.append(rew_sum)

        if (i + 1) % evalPeriod == 0:
            print(f'Episode {i}; Evaluation...')

            # evaluation
            eval_rews = []
            for _ in range(eval_episodes):
                _, eval_rew = run_episode(env, agent, train=False)
                eval_rews.append(eval_rew)
            mean_eval_rew = np.mean(eval_rews)
            print(f'Mean evaluation reward ({eval_episodes} episodes): {mean_eval_rew:.3f}')
            if mean_eval_rew > max_rew + 1.e-5:
                max_rew = mean_eval_rew
                print(f'New highest eval reward achieved!')
                agent.save('results/basic_dqn')

            rews_smoothed = moving_average(rews, 20)
            basic_plot(np.arange(len(losses)) + 1, np.array(losses),
                       xlabel='Episode', ylabel='loss',
                       plotFilePath='results/basic_dqn/losses.png',
                       save=True,)
            basic_plot(np.arange(len(losses)) + 1, np.log(np.array(losses)),
                       xlabel='Episode', ylabel='log(loss)',
                       plotFilePath='results/basic_dqn/log_losses.png',
                       save=True, )
            basic_plot(np.arange(len(rews_smoothed)) + 1, rews_smoothed,
                       xlabel='Episode', ylabel='reward, smoothed',
                       ylim=[-1., 1.],
                       plotFilePath='results/basic_dqn/reward.png',
                       save=True, )


if __name__ == '__main__':
    main()
