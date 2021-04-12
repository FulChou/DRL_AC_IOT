'''
Author: Ful Chou
Date: 2021-03-22 17:15:19
LastEditors: Ful Chou
LastEditTime: 2021-04-08 15:51:18
FilePath: /RL-Adventure-2/test.py
Description: What this document does
'''
import torch
import numpy as np
import gym

d =  {'step0': {'env0': [0.09354058414671297, 0.5865967145409838, -0.1952603898855059, -1.241520752959235], 'env1': [0.09354058414671297, 0.5865967145409838, -0.1952603898855059, -1.241520752959235]}, 
      'step1': {'env0': [0.04027106694159725, 0.022438353581838946, -0.02712520930711043, -0.003723000258639128], 'env1': [0.04027106694159725, 0.022438353581838946, -0.02712520930711043, -0.003723000258639128]},
      'step2': {'env0': [0.04071983401323403, -0.1722842928199783, -0.027199669312283216, 0.2802796879767939], 'env1': [0.04071983401323403, -0.1722842928199783, -0.027199669312283216, 0.2802796879767939]}, 
      'step3': {'env0': [0.03727414815683446, -0.3670078992520368, -0.021594075552747338, 0.5642613418868999], 'env1': [0.03727414815683446, -0.3670078992520368, -0.021594075552747338, 0.5642613418868999]},
      'step4': {'env0': [0.029933990171793726, -0.1715897273907912, -0.01030884871500934, 0.2648542598371409], 'env1': [0.029933990171793726, -0.1715897273907912, -0.01030884871500934, 0.2648542598371409]}
      }
def test():
    states = []
    for step, envs in d.items():
        step_states = []
        for env, value in envs.items():
            print('value',value)
            # res = [float(i) for i in value]
            # # print('new: ',res,type(res[0]))
            # step_states.append(res)
            step_states.append([float(i) for i in value])
            # print(step_states)
        states.append(step_states)
    print(states) 
    res = torch.FloatTensor(states)
    torch.set_printoptions(precision=20) # 只是打印不显示那么多位数。。
    print(res)
    # return torch.FloatTensor(states)



if __name__ == '__main__':
    gymenv = gym.make('Atlantis-v0')
    for i_episode in range(20):
        observation = gymenv.reset()
        for t in range(100):
            gymenv.render()
            # print(observation)
            action = gymenv.action_space.sample()
            observation, reward, done, info = gymenv.step(action)
            if done:
                print("Episode finished after {} timesteps and in {} episode".format(t+1,i_episode+1))
                break
    pass
