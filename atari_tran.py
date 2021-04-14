'''
Author: Ful Chou
Date: 2021-04-08 17:31:08
LastEditors: Ful Chou
LastEditTime: 2021-04-10 14:27:15
FilePath: /RL-Adventure-2/atari.py
Description: What this document does
'''
from time import time
import gym
import numpy as np
from itertools import count
from collections import namedtuple
from PIL import Image

import matplotlib.pyplot as plt

import torch
from torch.distributions.transforms import StackTransform
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torchsummary import summary
import json
import time

render = True
n_frames = 1
gamma = 0.99
learning_rate = 3e-2
log_interval = 10
N_episodes = 100
size = (82, 82)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


env = gym.make('Breakout-v0') # Breakout-v0 Atlantis-v0
# print(env.action_space)
# env1 = gym.make('AirRaid-ram-v0')
# print(env1.action_space )
# AirRaid-ram-v0

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        
        self.conv1 = nn.Conv2d(3 * n_frames, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 8, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(8)
        self.actor_head = nn.Linear(392, env.action_space.n)
        self.critic_head = nn.Linear(392, 1)

        self.episode_rewards = []
        self.episode_actions = []

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))) # 给了 2张（3，82，82）的图片？， 应为 summary 给了一帧图片！
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        action_prob = F.softmax(self.actor_head(x.view(x.size(0), -1)), dim=-1)
        critic_value = self.critic_head(x.view(x.size(0), -1))
        return action_prob, critic_value

model = ActorCritic().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
eps = np.finfo(np.float32).eps.item() # 非负最小值
# summary(model,input_size=(3,82,82))

def predict(actor_critic, state, new_probs, action):
    state = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0).float().to(device)
    probs, critic_value = actor_critic(state)
    probs.data = torch.Tensor(new_probs)
    m = Categorical(probs)
    action = torch.Tensor([action])
    actor_critic.episode_actions.append(SavedAction(m.log_prob(action), critic_value))
    return action.item()

def finish_episode():
    ''' compute loss, update model:
    '''
    R = 0
    episode_actions = model.episode_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.episode_rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards).to(device)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), r in zip(episode_actions, rewards):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.mse_loss(value[0], torch.tensor(r).to(device)))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.episode_rewards[:]
    del model.episode_actions[:]

def collect_frames(n_frames, action, frame_list=[]):
    ''' 输入动作， 收集之后的 n frame 图像：
    '''
    reward = 0
    for _ in range(n_frames):
        state, temp_reward, done, _ = env.step(action)
        reward += temp_reward
        state = Image.fromarray(state)
        state = np.array(state.resize(size, Image.ANTIALIAS), dtype=np.float32) # 图像滤波， 转换成 （82， 82， 3） 的图像
        frame_list.append(state)
    return np.concatenate(frame_list, axis=2), reward, done


# 不需要与环境进行交互，直接拿到 数据计算 loss ，更新模型即可：
def atari_state_model_params(data):
    start = time.time()
    probs_list = data['probs_list']
    states = data['states']
    actions = data['actions']
    # rewards = []
    # action_value_pairs = data['action_value_pairs']
    # print(len(probs_list),len(states), len(actions) )
    for i in range(len(states)):
        state = np.array(states[i])
        action = predict(model,state,probs_list[i],actions[i])
    model.episode_rewards = data['action_rewards']
    finish_episode()
    # print(model.state_dict())
    res = {k:v.tolist() for k, v in model.state_dict().items()}
    end = time.time()
    print('episode train time: {}'.format(end-start))
    return json.dumps(res)

    # TODO : 将 state 带入 model，按照网络参数更新模型计算的数据，然后 update模型,然后将模型参数回传

    # for i_episode in range(1, N_episodes + 1):
    #     state = env.reset()
    #     state = Image.fromarray(state)
    #     state = np.array(state.resize(size, Image.ANTIALIAS), dtype=np.float32) # 图像转换:
    #     if n_frames > 1: # 帧数 > 1
    #         state, reward, done = collect_frames(n_frames - 1, np.random.randint(
    #             env.action_space.n), frame_list=[state])
    #     else:
    #         done = False
    #     t = 0
    #     while not done  and t < 10000:
    #         t += 1
    #         action = select_action(model, state)
    #         state, reward, done = collect_frames(n_frames, action, frame_list=[])
    #         model.episode_rewards.append(reward)
    #         if render:
    #             env.render()
    #     # 更新对应的 值就行了，
    #     # update_episode_status()
    #     finish_episode()
    #     running_time = running_time * 0.99 + t * 0.01
    #     if i_episode % log_interval == 0:
    #         print('Episode {}\tLast Length: {:5d}\tAverage length: {:.2f}'.format(
    #             i_episode, t, running_time
    #         ))


def main():
    running_time = 0
    for i_episode in range(1, N_episodes + 1):
        state = env.reset()
        state = Image.fromarray(state)
        state = np.array(state.resize(size, Image.ANTIALIAS), dtype=np.float32) # 图像转换:
        if n_frames > 1: # 帧数 > 1
            state, reward, done = collect_frames(n_frames - 1, np.random.randint(
                env.action_space.n), frame_list=[state])
        else:
            done = False
        t = 0
        while not done  and t < 10000:
            t += 1
            action = select_action(model, state)
            state, reward, done = collect_frames(n_frames, action, frame_list=[])
            model.episode_rewards.append(reward)
            if render:
                env.render()

        finish_episode()
        running_time = running_time * 0.99 + t * 0.01
        if i_episode % log_interval == 0:
            print('Episode {}\tLast Length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_time
            ))



if __name__ == '__main__':
    main()