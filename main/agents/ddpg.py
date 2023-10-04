import numpy as np
import random 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class DDQNImproved:
    def __init__(self, 
                 base_model: nn.Module, 
                 critic_model: nn.Module,
                 replay_buffer: ReplayMemory, 
                 gamma: float, 
                 lr: float,
                 weight_decay: float, 
                 tau: float, 
                 batch_size: int,
                 step=1,
                 n_actions=15, 
                 loss_fct = nn.MSELoss(), 
                 upper_bound = 2,
                 lower_bound = 0,
                 noise_part = 0, 
                 theta=0.1,
                 mu=0.05,
                 dt=0.001,
                 std=0.2) -> None:
        # Buffer
        self.buffer = replay_buffer

        # Actor
        self.actor = base_model
        self.target_actor = deepcopy(base_model)
        self.actor.output_function = torch.sigmoid
        self.optimizer = optim.AdamW(self.actor.parameters(),
                                     lr=lr,
                                     amsgrad=True,
                                     weight_decay=weight_decay)
        # Critic
        self.critic = deepcopy(critic_model)
        self.target_critic = deepcopy(critic_model)
        self.optimizer_critic = optim.AdamW(self.critic.parameters(),
                                            lr=lr*1.05,
                                            amsgrad=True,
                                            weight_decay=weight_decay)
        
        # Params
        self.tau =tau
        self.gamma = np.float32(gamma)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.criterion = loss_fct
        self.noise_part = noise_part
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.std = std
        self.upper_bound = np.float32(upper_bound)
        self.lower_bound = np.float32(lower_bound)
        self.step = step
        
        # Initialize nets
        self.actor.apply(self._init_weights)
        self.target_actor.load_state_dict(self.actor.state_dict())

        # Initialize nets
        self.critic.apply(self._init_weights)
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _init_weights(self, lay):
        if isinstance(lay, nn.Linear):
            torch.nn.init.xavier_normal_(lay.weight)
            lay.bias.data.fill_(1.0)

    def noise_process(self, prestep):
        # step_noise =  (prestep +
        #                self.theta * (self.mu-prestep) * np.exp(-self.dt) +
        #                self.std * np.sqrt(self.dt) * np.random.normal())
        
        # step_noise = step_noise*np.exp(-self.step*0.00001)
        step_noise = np.random.normal(loc=0.0, scale=np.exp(-self.step*self.dt))
                                     
        return step_noise

    def sample_action(self, state, training=True):
        action_choice = self.actor(state)*(self.upper_bound)
        if training:
            self.noise_part = self.noise_process(self.noise_part)
            action_choice += self.noise_part

        action_ = torch.clamp(action_choice, min=self.lower_bound)

        return action_

    def update_target(self):
        model_states = self.actor.state_dict()
        target_states = self.target_actor.state_dict()

        for key in model_states:
            target_states[key] = self.tau*model_states[key] + (1-self.tau)*target_states[key]
        self.target_actor.load_state_dict(target_states)

        model_states = self.critic.state_dict()
        target_states = self.target_critic.state_dict()

        for key in model_states:
            target_states[key] = self.tau*model_states[key] + (1-self.tau)*target_states[key]
        self.target_critic.load_state_dict(target_states)

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
        with torch.no_grad():
            self.target_actor.eval()
            self.target_critic.eval()

            target_actions = self.target_actor(next_state_s)
            target_critic_action = self.target_critic(torch.cat((next_state_s, target_actions), 1))

            self.target_actor.train()
            self.target_critic.train()

        critic_value = self.critic(torch.cat((state_s, action_s), 1)).squeeze()
        target_value = self.gamma*target_critic_action.squeeze() + reward_s.squeeze()
        critic_loss = self.criterion(critic_value, target_value)

        self.optimizer_critic.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), 
                                        100)
        self.optimizer_critic.step()

        policy_action = self.actor(state_s)
        actor_loss = -self.critic(torch.cat((state_s, policy_action), 1))
        actor_loss = actor_loss.mean()

        self.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self.actor.parameters(), 
                                        100)
        self.optimizer.step()
