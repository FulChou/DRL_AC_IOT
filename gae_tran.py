
import math
import random
import gym
import numpy as np

import torch
from torch import tensor
import torch.nn as nn
from torch.nn import parameter
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Normal
import matplotlib.pyplot as plt
import time
from common.multiprocessing_env import SubprocVecEnv
from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def make_env():  # why?
    def _thunk():
        env = gym.make(env_name)
        return env
    return _thunk

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, reduce_factor, up_factor, std=0.0):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size // reduce_factor),
            nn.ReLU(),
            nn.Linear(hidden_size // reduce_factor,
                      hidden_size // reduce_factor),
            nn.ReLU(),
            nn.Linear(hidden_size // reduce_factor, num_outputs),
            nn.Softmax(dim=1),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        self.apply(init_weights)

    def forward(self, x):
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist



class Critic(nn.Module):
    def __init__(self,  num_inputs, num_outputs, hidden_size, reduce_factor, up_factor, std=0.0):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size*up_factor),
            nn.ReLU(),
            nn.Linear(hidden_size*up_factor, hidden_size*up_factor),
            nn.ReLU(),
            nn.Linear(hidden_size*up_factor, 1)
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        self.apply(init_weights)

    def forward(self, x):
        return self.critic(x)


def plot(frame_idx, rewards):
    # clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    # plt.savefig('frame %s. reward: %s.jpg' % (frame_idx, rewards[-1]))
    plt.show()


def test_env(vis=False):
    state = env.reset()
    if vis:
        env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist = actor(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis:
            env.render()
        total_reward += reward
    return total_reward


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def dict2list(states_dic):  # 2 list
    states = []
    for step, envs in states_dic.items():
        step_states = []
        for env, value in envs.items():
            step_states.append(value)
        states.append(list(step_states))
    return states


gae_env_param = {
    'num_envs': 8,
    'env_name': "Pendulum-v0", # baseline 最差: -3254.72/2  均值：1627 
    'hidden_size': 256,
    'lr': 3e-2,
    'num_steps': 20,
    'reduce_factor': 1,
    'up_factor': 3,
    'stop_max': 1,
    'target': 500,
    'max_frames':100000,
}

# hyper env params
num_envs = gae_env_param['num_envs']
env_name = gae_env_param['env_name']
env = gym.make(env_name)
envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)
num_inputs = envs.observation_space.shape[0]
num_outputs = envs.action_space.shape[0]
# model Hyper params: baseline, mlp Multilayer Perceptron 64 2 layer
hidden_size = gae_env_param['hidden_size']
lr = gae_env_param['lr']
num_steps = gae_env_param['num_steps']
reduce_factor = gae_env_param['reduce_factor']
up_factor = gae_env_param['up_factor']
actor = Actor(num_inputs, num_outputs, hidden_size,
              reduce_factor, up_factor).to(device)
critic = Critic(num_inputs, num_outputs, hidden_size,
                reduce_factor, up_factor).to(device)
optimizer = optim.Adam(actor.parameters(), lr=lr)
optimizer2 = optim.Adam(critic.parameters(), lr=lr)
# summary(actor, input_size=(1,4))
# summary(critic, input_size=(1,4))

def gae_update_model(status):
    states = dict2list(status['states'])
    actions = dict2list(status['actions'])
    pre_rewards = dict2list(status['rewards'])
    pre_masks = dict2list(status['masks'])
    mu_s =  dict2list(status['mu_s'])
    std_s = dict2list(status['std_s'])
    max_frames = 20
    frame_idx = 0
    start = time.time()
# train：
    while frame_idx < max_frames:
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0

        for i in range(num_steps):
            state = torch.FloatTensor(states[i]).to(device)
            dist, value = actor(state), critic(state)  # 运行 actor - critic网络模型
            # dist.probs.data = torch.FloatTensor(dist_probs[i])
            # print(dist_probs[i])
            dist.loc.data = torch.FloatTensor(mu_s[i])
            dist.scale.data = torch.FloatTensor(std_s[i])

            action = dist.sample()
            action.data = torch.FloatTensor(actions[i])
            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)
            entropy += dist.entropy().mean()
            values.append(value)
            rewards.append(torch.FloatTensor(
                pre_rewards[i]).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(
                pre_masks[i]).unsqueeze(1).to(device))
            frame_idx += 1

        next_value = critic(torch.FloatTensor(states[-1]))
        log_probs = torch.cat(log_probs)  # 按照行拼接
        returns = compute_gae(next_value, rewards, masks,values) #
        returns = torch.cat(returns).detach()
        values = torch.cat(values)
        advantage = returns - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        optimizer.zero_grad()
        optimizer2.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer2.step()
    model_dict = actor.state_dict()
    parameters = {k: v.tolist() for k, v in model_dict.items() if 'actor' in k}
    end = time.time()
    print('time: ', end - start)
    return parameters
