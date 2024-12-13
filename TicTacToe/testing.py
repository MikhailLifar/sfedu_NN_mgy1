from util import *
from TitTacToe import *


def interactive_test_0():
    env = TicTacToe(size=10, win_size=3)
    renderer = CLIRenderer(env)

    renderer.render()
    op = 'start'
    while op != 'q':
        op = input('Enter your move in (x y) format >>> ')
        try:
            x, y = [int(el) for el in op.split()]
            code = env.move(x, y)
            if code < 0:
                raise ValueError
            renderer.render()
            if code == 1:
                print('We have a winner! Game is reset now')
                env.reset()
        except ValueError:
            print('Incorrect move. Try again!')
            continue


def interactive_test_1():
    tictactoe = TicTacToe(size=3, win_size=3)
    renderer = CLIRenderer(tictactoe)
    env = TicTacToeEnv(tictactoe, RandomPlayer(tictactoe.size * tictactoe.size), renderer)

    env.reset()
    env.render()
    op = 'start'
    while op != 'q':
        op = input('Enter the action in y * win_size + x format >>> ')
        try:
            action = int(op)
            obs, rew, done, _, _ = env.step(action)
            env.render()
            if done:
                print('The end of the game is reached. The environment is reset')
                env.reset()
        except ValueError:
            print('Incorrect move. Try again!')
            continue


def env_test_0():
    tictactoe = TicTacToe(3, 3)
    renderer = CLIRenderer(tictactoe)
    env = TicTacToeEnv(tictactoe, RandomPlayer(tictactoe.size * tictactoe.size),
                       renderer=renderer)
    player = RandomPlayer(tictactoe.size * tictactoe.size)

    np.random.seed(1)
    for i in range(2):
        print('-' * 50)
        if i == 0:
            print('Agent plays for "X"')
        else:
            print('Agent plays for "0"')
        obs, _ = env.reset()
        print(f'Observation 0:\n{obs}')
        env.render()
        done = False
        j = 1
        while np.any(env.tictactoe.field == 0):
            act = player.act(obs, {})
            obs, rew, done, _, _ = env.step(act)
            print(f'Observation {j}:\n{obs}')
            print(f'Reward {j}:\n{rew}')
            env.render()
            j += 1


# def env_test_1():
#     tictactoe = TicTacToe(3, 3)
#     renderer = CLIRenderer(tictactoe)
#     env = TicTacToeEnv(tictactoe, RandomPlayer(tictactoe.size * tictactoe.size),
#                        renderer=renderer)
#     player1 = RandomPlayer(tictactoe.size * tictactoe.size)
#     # player2 = RandomOpponent(tictactoe.size * tictactoe.size)
#
#     np.random.seed(1)
#     for i in range(2):
#         print('-' * 50)
#         if i == 0:
#             print('Agent plays for "X"')
#         else:
#             print('Agent plays for "0"')
#         obs, info = env.reset()
#         print(f'Observation 0:\n{obs}')
#         env.render()
#         done = False
#         j = 1
#         while done:
#             act = player1.act(obs, info)
#             obs, rew, done, _, _ = env.step(act)
#             print(f'Observation {j}:\n{obs}')
#             print(f'Reward {j}:\n{rew}')
#             env.render()
#             j += 1


def env_test_2():
    tictactoe = TicTacToe(3, 3)
    renderer = CLIRenderer(tictactoe)

    player1 = UnconditionalPlayer(
        [0, 1, 2, 3],
    )
    player2 = UnconditionalPlayer(
        [5, 8, 4, 7],
    )

    env = TicTacToeEnv(tictactoe, player2,
                       renderer=renderer)

    np.random.seed(1)
    for i in range(2):
        print('-' * 50)
        if i == 0:
            print('Agent plays for "X"')
        else:
            print('Agent plays for "0"')
        obs, info = env.reset()
        print(f'Observation 0:\n{obs}')
        env.render()
        done = False
        j = 1
        while not done:
            act = player1.act(obs, info)
            obs, rew, done, _, _ = env.step(act)
            print(f'Observation {j}:\n{obs}')
            print(f'Reward {j}:\n{rew}')
            env.render()
            j += 1
        player1.reset()
        player2.reset()


def main():
    # interactive_test_0()
    # interactive_test_1()
    # env_test_0()
    env_test_2()


if __name__ == '__main__':
    main()
