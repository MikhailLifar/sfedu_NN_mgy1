import os
import sys

from util import *
from ReplayBuffer import ReplayBuffer
from agents import dqn
from TitTacToe import *

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

    width = 32
    d = 3
    body = []
    for _ in range(d - 1):
        body.extend([
            nn.Linear(width, width),
            nn.ReLU(),
        ])
    qnet = nn.Sequential(
        nn.Linear(fieldSize, width),
        nn.ReLU(),
        *body,
        nn.Linear(width, fieldSize),
    )
    optimizer = optim.Adam(qnet.parameters(), lr=3.e-4, weight_decay=1.e-4)
    buffer = ReplayBuffer(fieldSize, 1, capacity=10_000)
    agent = dqn.DQN(qnet, optimizer, buffer, fieldSize,
                    eps_end=0.01, eps_decay=0.999)
    # agent = dqn.DDQN(qnet, optimizer, buffer, fieldSize,
    #                  eps_end=0.1)
    resDir = 'results/basic_dqn'
    # resDir = 'results/basic_ddqn'
    os.makedirs(resDir, exist_ok=True)

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
                agent.save(resDir)

            rews_smoothed = moving_average(rews, 20)
            basic_plot(np.arange(len(losses)) + 1, np.array(losses),
                       xlabel='Episode', ylabel='loss',
                       plotFilePath=f'{resDir}/losses.png',
                       save=True,)
            basic_plot(np.arange(len(losses)) + 1, np.log(np.array(losses)),
                       xlabel='Episode', ylabel='log(loss)',
                       plotFilePath=f'{resDir}/log_losses.png',
                       save=True, )
            basic_plot(np.arange(len(rews_smoothed)) + 1, rews_smoothed,
                       xlabel='Episode', ylabel='reward, smoothed',
                       ylim=[-1., 1.],
                       plotFilePath=f'{resDir}/reward.png',
                       save=True, )


if __name__ == '__main__':
    main()
