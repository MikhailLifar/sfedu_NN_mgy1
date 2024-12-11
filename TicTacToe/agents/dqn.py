import copy

import numpy as np
import torch
import torch.nn.functional as nnf


class DQN:
    def __init__(self, qnet, optimizer, buffer,
                 nActions,
                 batch_size=256,
                 gamma=0.99,
                 eps_start=1.,
                 eps_end=0.01,
                 eps_decay=0.99,
                 tau=0.01):
        self.qnet = qnet
        self.target_qnet = copy.deepcopy(qnet)
        self.optimizer = optimizer
        self.buffer = buffer
        self.batch_size = batch_size

        self.nActions = nActions
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau

        self.train_mode = True

    def set_train(self):
        self.train_mode = True

    def set_eval(self):
        self.train_mode = False

    def act(self, obs, info=None):
        if self.train_mode and (np.random.random() < self.eps):
            return np.random.choice(self.nActions)
        with torch.no_grad():
            qs = self.qnet(torch.Tensor(obs)).cpu().numpy()
            if info is not None and 'action_mask' in info:
                qs[~info['action_mask']] = -np.inf
            action = np.argmax(qs)
            return action

    def store(self, state, action, reward, next_state, done):
        self.buffer.store(state, action, reward, next_state, done)

    def update_target_net(self):
        target_state_dict = self.target_qnet.state_dict()
        current_state_dict = self.qnet.state_dict()
        for key in target_state_dict:
            target_state_dict[key] = (1. - self.tau) * target_state_dict[key] + self.tau * current_state_dict[key]
        self.target_qnet.load_state_dict(target_state_dict)

    def train(self):
        if self.buffer.get_size() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.Tensor(states)
        actions = torch.Tensor(actions).long()
        rewards = torch.Tensor(rewards)
        next_states = torch.Tensor(next_states)
        dones = torch.Tensor(dones)

        with torch.no_grad():
            not_done = (1. - dones.float())
            next_q, _ = self.target_qnet(next_states).max(dim=1)
            next_q = next_q.unsqueeze(1)
            target = rewards + not_done * self.gamma * next_q

        self.qnet.train()
        self.optimizer.zero_grad()
        current = self.qnet(states).gather(1, actions)
        loss = nnf.huber_loss(current, target)
        loss.backward()
        self.optimizer.step()

        self.update_target_net()

        self.eps = max(self.eps * self.eps_decay, self.eps_end)

        return float(loss.item())

    def save(self, dirPath):
        torch.save(self.qnet.state_dict(), f'{dirPath}/qnet.pt')
        torch.save(self.target_qnet.state_dict(), f'{dirPath}/target_qnet.pt')

    def load(self, dirPath):
        self.qnet.load_state_dict(torch.load(f'{dirPath}/qnet.pt', weights_only=True))
        self.target_qnet.load_state_dict(torch.load(f'{dirPath}/target_qnet.pt', weights_only=True))


class DDQN(DQN):

    def train(self):
        if self.buffer.get_size() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.Tensor(states)
        actions = torch.Tensor(actions).long()
        rewards = torch.Tensor(rewards)
        next_states = torch.Tensor(next_states)
        dones = torch.Tensor(dones)

        with torch.no_grad():
            not_done = (1. - dones.float())
            _, next_q_idx = self.qnet(next_states).max(dim=1)
            next_q_idx = next_q_idx.long().unsqueeze(1)
            next_q = self.target_qnet(next_states).gather(1, next_q_idx)
            next_q = next_q.unsqueeze(1)
            target = rewards + not_done * self.gamma * next_q

        self.qnet.train()
        self.optimizer.zero_grad()
        current = self.qnet(states).gather(1, actions)
        loss = nnf.huber_loss(current, target)
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.qnet.parameters(), 100)
        self.optimizer.step()

        self.update_target_net()

        self.eps = max(self.eps * self.eps_decay, self.eps_end)

        return float(loss.item())
