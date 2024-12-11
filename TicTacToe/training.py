import os
import sys
from itertools import cycle

import numpy as np

from util import *
from ReplayBuffer import ReplayBuffer
from agents import dqn
from TitTacToe import *

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.expanduser('~/SupervisedML/repos'))
from supervised_ml.plot_util import *


def run_episode(env, agent, train_mode=True):
    if train_mode:
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
            act = agent.act(obs)
            next_obs, rew, done, _, _ = env.step(act)
            ep_rews.append(rew)
            obs = next_obs
        return None, np.sum(ep_rews)


def train():
    tictactoe = TicTacToe(3, 3)
    # tictactoe = TicTacToe(13, 5)
    renderer = CLIRenderer(tictactoe)
    fieldSize = tictactoe.size * tictactoe.size
    env = TicTacToeEnv(tictactoe, RandomOpponent(fieldSize), renderer)

    width = 64
    d = 5
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
    buffer = ReplayBuffer(fieldSize, 1, capacity=30_000)
    agent = dqn.DQN(qnet, optimizer, buffer, fieldSize,
                    eps_end=0.01, eps_decay=0.999)
    # agent = dqn.DDQN(qnet, optimizer, buffer, fieldSize,
    #                  eps_end=0.01, eps_decay=0.999)
    # resDir = 'results/basic_dqn'
    resDir = 'results/dqn_3x3'
    # resDir = 'results/dqn_13x5'
    # resDir = 'results/basic_ddqn'
    # resDir = 'results/ddqn_3x3'
    os.makedirs(resDir, exist_ok=True)

    n_episodes = 100_000
    # n_episodes = 200
    evalPeriod = 1000
    eval_episodes = 100
    max_rew = -np.inf
    losses = []
    rews = []
    for i in range(n_episodes):
        agent.set_train()
        loss_mean, rew_sum = run_episode(env, agent)
        if loss_mean is not None:
            losses.append(loss_mean)
        rews.append(rew_sum)

        if (i + 1) % evalPeriod == 0:
            print(f'Episode {i}; Evaluation...')

            # evaluation
            agent.set_eval()
            eval_rews = []
            for _ in range(eval_episodes):
                _, eval_rew = run_episode(env, agent, train_mode=False)
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


def benchmark(player1, player2, env, n_games=1000):
    """
    player: tuple(player_id, player_obj)
    :param env:
    :param n_games:
    :param player1:
    :param player2:
    :return:
    """

    assert env.agent_turn == 1
    assert env.agent_turn_fixed

    id1, player1 = player1
    id2, player2 = player2

    counter = {
        f'{id1} for "X"': 0,
        f'{id1} for "0"': 0,
        f'{id2} for "X"': 0,
        f'{id2} for "0"': 0,
    }

    ids = [id1, id2]
    players = [player1, player2]

    for _ in range(2):
        player1, player2 = players
        id1, id2 = ids
        for _ in range(n_games):
            done = False
            obs, info = env.reset()
            while not done:
                act = player1.act(obs, info)
                obs, code, info = env.move(act)
                done = (code == 1) or (not np.any(info['action_mask']))
                if code == 1:
                    counter[f'{id1} for "X"'] += 1
                if done:
                    break
                act = player2.act(obs, info)
                obs, code, info = env.move(act)
                done = (code == 1) or (not np.any(info['action_mask']))
                if code == 1:
                    counter[f'{id2} for "0"'] += 1
        players = players[::-1]
        ids = ids[::-1]

    return counter


def main():
    # model benchmarking
    tictactoe = TicTacToe(3, 3)
    fieldSize = tictactoe.size * tictactoe.size
    env = TicTacToeEnv(tictactoe, RandomOpponent(fieldSize), None,
                       agent_turn=1, agent_turn_fixed=True)

    agentDir = 'results/dqn_3x3'
    width = 64
    d = 5
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
    qnet.load_state_dict(torch.load(f'{agentDir}/qnet.pt', weights_only=True))

    optimizer = optim.Adam(qnet.parameters(), lr=3.e-4, weight_decay=1.e-4)
    buffer = ReplayBuffer(fieldSize, 1, capacity=30_000)
    agent = dqn.DQN(qnet, optimizer, buffer, fieldSize,
                    eps_end=0.01, eps_decay=0.999)

    player1 = ('DQN', agent)
    player2 = ('Random', RandomOpponent(fieldSize))

    results = benchmark(player1, player2, env)
    print(results)

    basic_barplot(results, plotFilePath=f'{agentDir}/benchmark.png')


if __name__ == '__main__':
    # train()
    main()
