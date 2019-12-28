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
from utils import set_global_seeds

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", nargs='?')
parser.add_argument("--num_timesteps", default=100000, nargs='?', type=int)
parser.add_argument("--env", default="TwoBumps-v1", nargs='?')
parser.add_argument("--seed", default=0, nargs='?', type=int)
parser.add_argument("--final_epsilon", default=0.02, nargs='?')
parser.add_argument("--exploration_fraction", default=0.1, nargs='?')
parser.add_argument("--replay_size", default=50000, nargs='?', type=int)
parser.add_argument("--batch_size", default=32, nargs='?')
parser.add_argument("--target_network_update_freq", default=500, nargs='?', type=int)
parser.add_argument("--num_prepopulate_episode", default=1000, nargs='?', type=int)
parser.add_argument("--print_freq", default=100, nargs='?', type=int)
parser.add_argument("--model_save_freq", default=10000, nargs='?', type=int)
parser.add_argument("--train_freq", default=1, nargs='?', type=int)
parser.add_argument("--gamma", default=0.99, nargs='?', type=float)
parser.add_argument("--episode_training_len", default=8, nargs='?', type=int)


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
train_dqn(
    env,
    args.num_timesteps,
    replay_size=args.replay_size,
    batch_size=args.batch_size,
    exploration=exploration,
    gamma=args.gamma,
    train_freq=args.train_freq,
    print_freq=args.print_freq,
    model_save_freq=args.model_save_freq,
    target_network_update_freq=args.target_network_update_freq,
    num_prepopulate_episode=args.num_prepopulate_episode, 
    episode_training_len=args.episode_training_len
)