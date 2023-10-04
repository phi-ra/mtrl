import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from copy import deepcopy
from .utils import Transition, ReplayMemory

class DQNAgent:
    def __init__(self, 
                 base_model: nn.Module, 
                 replay_buffer: ReplayMemory, 
                 gamma: float, 
                 lr: float,
                 weight_decay: float, 
                 tau: float, 
                 batch_size: int,
                 n_actions=15, 
                 loss_fct = nn.SmoothL1Loss()) -> None:
        self.policy_net = base_model
        self.target_net = deepcopy(base_model)
        self.buffer = replay_buffer
        self.optimizer = optim.AdamW(self.policy_net.parameters(),
                                     lr=lr,
                                     amsgrad=True,
                                     weight_decay=weight_decay)
        self.tau =tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.criterion = loss_fct
        
        # Initialize nets
        self.policy_net.apply(self._init_weights)
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _init_weights(self, lay):
        if isinstance(lay, nn.Linear):
            torch.nn.init.xavier_normal_(lay.weight)
            lay.bias.data.fill_(1.0)

    def sample_action(self, state, epsilon):
        if np.random.uniform() > epsilon:
            self.policy_net.eval()
            with torch.no_grad():
                action_idx = self.policy_net(state).argmax()
            self.policy_net.train()
            return action_idx
        else:
            return torch.tensor(random.choices(range(self.n_actions), k=1)[0])

    def update_target(self):
        model_states = self.policy_net.state_dict()
        target_states = self.target_net.state_dict()

        for key in model_states:
            target_states[key] = self.tau*model_states[key] + (1-self.tau)*target_states[key]
        self.target_net.load_state_dict(target_states)

    def update_policy(self):
        if len(self.buffer) < self.batch_size:
            return
        sample = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*sample))

        state_s = torch.cat(batch.state)
        reward_s = torch.cat(batch.reward)
        action_s = torch.cat(batch.action)
        next_state_s = torch.cat(batch.next_state)

        # Get updates
        q_values_hat = self.policy_net(state_s).gather(1, action_s).squeeze()
        
        with torch.no_grad():
            self.target_net.eval()
            next_q_values, _ = self.target_net(next_state_s).max(1)
            self.target_net.train()

        target_value = self.gamma * next_q_values.squeeze() + reward_s.squeeze()

        policy_loss = self.criterion(q_values_hat, target_value)
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(),
                                        100)
        self.optimizer.step()
