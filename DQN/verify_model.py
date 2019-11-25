import torch.nn as nn
import torch
import gym
import torch.nn.functional as F
from model import DQN


model = DQN()
model.load_state_dict(torch.load("model-4.pt"))

env = gym.make("TwoBumps-v0")

state = env.reset()
env.render()

total_sim = 100
cnt_sim = 0
ep_reward = 0
cnt_reward = 0

while cnt_sim < total_sim:
    action = model(torch.tensor(state, dtype=torch.float)).argmax().item()
    state, reward, done, _ = env.step(action)
    ep_reward += reward
    env.render()

    if done:
        cnt_sim += 1
        print(ep_reward)

        if ep_reward > 0:
            cnt_reward += 1

        ep_reward = 0
        state = env.reset()

print("Test success rate: ", cnt_reward/total_sim)
env.close()