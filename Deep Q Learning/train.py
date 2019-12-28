#!/usr/bin/env python
# coding: utf-8

import gym
import numpy as np
import sys
import os

import torch
from datetime import datetime
from baselines import logger
from exploration import LinearSchedule
from model import train_dqn
from utils import *

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", nargs='?')
parser.add_argument("--num_timesteps", default=100000, nargs='?', type=int)
parser.add_argument("--env", default="TwoBumps-v0", nargs='?')
parser.add_argument("--seed", default=0, nargs='?', type=int)
parser.add_argument("--final_epsilon", default=0.02, nargs='?')
parser.add_argument("--exploration_fraction", default=0.6, nargs='?')
parser.add_argument("--replay_size", default=50000, nargs='?', type=int)
parser.add_argument("--batch_size", default=32, nargs='?')
parser.add_argument("--target_network_update_freq", default=500, nargs='?', type=int)
parser.add_argument("--t_learning_start", default=1000, nargs='?', type=int)
parser.add_argument("--print_freq", default=50, nargs='?', type=int)
parser.add_argument("--train_freq", default=1, nargs='?', type=int)
parser.add_argument("--gamma", default=1.0, nargs='?', type=float)


args = parser.parse_args()

env = gym.make(args.env)
set_global_seeds(args.seed)

# Exploration schedule         
exploration = LinearSchedule(1.0, args.final_epsilon, int(args.num_timesteps * args.exploration_fraction))

# Logging
today = datetime.today()
date = today.strftime('%d-%m-%Y')
time = today.strftime("%H:%M:%S")
logger.configure("~/logs/" + date + "/" + time + "/")

if args.model_path is None:
    dir = "~/models/" + date + "/" + time + "/"
else:
    dir = args.model_path

dir = os.path.expanduser(dir)
os.makedirs(os.path.expanduser(dir), exist_ok=True)

# Training
dqn_models, returns, lengths, losses = train_dqn(
    env,
    args.num_timesteps,
    replay_size=args.replay_size,
    batch_size=args.batch_size,
    exploration=exploration,
    gamma=args.gamma,
    train_freq=args.train_freq,
    print_freq=args.print_freq,
    target_network_update_freq=args.target_network_update_freq,
    t_learning_start=args.t_learning_start
)

# Save model
torch.save(dqn_models.state_dict(), dir + "model.pt")