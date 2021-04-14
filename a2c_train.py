
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

    def forward(self, x):
        probs = self.actor(x)
        dist = Categorical(probs)
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


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def dict2list(states_dic):  # 2 list
    states = []
    for step, envs in states_dic.items():
        step_states = []
        for env, value in envs.items():
            step_states.append(value)
        states.append(list(step_states))
    return states


a2c_env_param = {
    'num_envs': 2,
    'env_name': "CartPole-v0",
    'hidden_size': 64,
    'lr': 1e-3,
    'num_steps': 5,
    'reduce_factor': 1,
    'up_factor': 2,
    'stop_max': 3,
    'target': 195,
}

# hyper env params
num_envs = a2c_env_param['num_envs']
env_name = a2c_env_param['env_name']
env = gym.make(env_name)
envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)
num_inputs = envs.observation_space.shape[0]
num_outputs = envs.action_space.n
# model Hyper params: baseline, mlp Multilayer Perceptron 64 2 layer
hidden_size = a2c_env_param['hidden_size']
lr = a2c_env_param['lr']
num_steps = a2c_env_param['num_steps']
reduce_factor = a2c_env_param['reduce_factor']
up_factor = a2c_env_param['up_factor']
actor = Actor(num_inputs, num_outputs, hidden_size,
              reduce_factor, up_factor).to(device)
critic = Critic(num_inputs, num_outputs, hidden_size,
                reduce_factor, up_factor).to(device)
optimizer = optim.Adam(actor.parameters(), lr=lr)
optimizer2 = optim.Adam(critic.parameters(), lr=lr)
# summary(actor, input_size=(1,4))
# summary(critic, input_size=(1,4))

def get_status_return_model_parameters(status):
    states = dict2list(status['states'])
    actions = dict2list(status['actions'])
    pre_rewards = dict2list(status['rewards'])
    pre_masks = dict2list(status['masks'])
    dist_probs = dict2list(status['dist_probs'])
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
            dist.probs.data = torch.FloatTensor(dist_probs[i])
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
        returns = compute_returns(next_value, rewards, masks)
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
