import numpy as np


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=10_000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity = capacity

        self.states = np.zeros((self.capacity, self.state_dim))
        self.actions = np.zeros((self.capacity, self.action_dim))
        self.rewards = np.zeros((self.capacity, 1))
        self.next_states = np.zeros((self.capacity, self.state_dim))
        self.dones = np.zeros((self.capacity, 1))

        self.pointer = 0
        self.full = False

    def store(self, state, action, reward, next_state, done):
        p = self.pointer
        self.states[p] = state
        self.actions[p] = action
        self.rewards[p] = reward
        self.next_states[p] = next_state
        self.dones[p] = done

        self.pointer += 1
        if self.pointer == self.capacity:
            self.full = True
            self.pointer = 0

    def get_size(self):
        if self.full:
            return self.capacity
        return self.pointer

    def sample(self, sample_size):
        upper_bound_idx = self.capacity if self.full else self.pointer
        idx = np.random.choice(upper_bound_idx, sample_size, replace=False)
        return self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx]
