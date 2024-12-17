import copy
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

setPlotDefaults()


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
    env = TicTacToeEnv(tictactoe, [RandomPlayer(fieldSize), ], renderer)
    eval_env = copy.deepcopy(env)

    width = 32
    d = 1
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
    optimizer = optim.Adam(qnet.parameters(), lr=1.e-3, weight_decay=1.e-4)
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
                _, eval_rew = run_episode(eval_env, agent, train_mode=False)
                eval_rews.append(eval_rew)
            mean_eval_rew = np.mean(eval_rews)
            print(f'Mean evaluation reward ({eval_episodes} episodes): {mean_eval_rew:.3f}')
            if mean_eval_rew > max_rew + 1.e-5:
                # if (mean_eval_rew > 0.7) and (mean_eval_rew - max_rew > 0.1):
                #     print('New opponent added!')
                #     new_opponent = copy.deepcopy(agent)
                #     new_opponent.eps = 0.1
                #     new_opponent.eps_end = 0.1
                #     env.add_opponent(new_opponent)
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


# def check_q_func(qnet):
#     fieldSize = 3
#     tictactoe = TicTacToe(fieldSize, 3)
#     env = TicTacToeEnv(tictactoe,
#                        opponents=[RandomPlayer(fieldSize * fieldSize)],
#                        renderer=None)
#
#     def get_obs():
#         return env.agent_turn * env.tictactoe.field.flatten()
#
#     def get_pred(obs):
#         return qnet(torch.Tensor(obs)).cpu().detach().numpy()
#
#     obs0 = get_obs()
#     q0 = get_pred(obs0).reshape(fieldSize, fieldSize)
#     basic_map()


def main():
    # model benchmarking
    tictactoe = TicTacToe(3, 3)
    fieldSize = tictactoe.size * tictactoe.size
    env = TicTacToeEnv(tictactoe, [RandomPlayer(fieldSize), ], None,
                       agent_turn=1, agent_turn_fixed=True)

    agentDir = 'results/dqn_3x3'
    width = 32
    d = 1
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
                    eps_end=0.1, eps_decay=0.999, tau=1.e-2)
    agent.set_eval()

    player1 = ('DQN', agent)
    player2 = ('Random', RandomPlayer(fieldSize))

    results = benchmark(player1, player2, env)
    print(results)

    basic_barplot(results, plotFilePath=f'{agentDir}/benchmark.png')


def replot():
    agentDir = 'results/dqn_3x3_32x3_45k'
    plotData = readParams(f'{agentDir}/benchmark_data')[1]['0']
    print(plotData)

    x = ['wins', 'draws', 'losses']
    y = [plotData['DQN for "X"'],
         1000 - plotData['DQN for "X"'] - plotData['Random for "0"'],
         plotData['Random for "0"']]
    basic_bar(x, y, color=['g', 'b', 'r'], annot=dict(fontsize=22),
              ylim=[None, 1100],
              title='Trained agent benchmark, agent playing "X"',
              plotFilePath='results/final_results_X.png')

    x = ['wins', 'draws', 'losses']
    y = [plotData['DQN for "0"'],
         1000 - plotData['DQN for "0"'] - plotData['Random for "X"'],
         plotData['Random for "X"']]
    basic_bar(x, y, color=['g', 'b', 'r'], annot=dict(fontsize=22),
              ylim=[None, 1100],
              title='Trained agent benchmark, agent playing "0"',
              plotFilePath='results/final_results_0.png')


if __name__ == '__main__':
    # train()
    # main()
    replot()
