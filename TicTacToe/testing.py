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
    env = TicTacToeEnv(tictactoe, RandomOpponent(tictactoe.size * tictactoe.size), renderer)

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


def main():
    # interactive_test_0()
    interactive_test_1()


if __name__ == '__main__':
    main()
