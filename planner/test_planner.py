import torch
import numpy as np
import argparse
import tqdm
import sys
sys.path.append('../')
from arguments_ddpg import *
import cv2
import gym
import os
from torch.distributions import Categorical
from goal_env.recorder import play
from algos.ddpg_agent import ddpg_agent
from train import get_env_params
from planner.goal_plan import Planner
from tensorboardX import SummaryWriter
from datetime import datetime
from algos.networks import *

IMAGE_SIZE = 512

def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env-name', type=str, default='AntMaze-v1', help='the environment name')
    parser.add_argument('--test', type=str, default='AntMazeTest-v1')
    parser.add_argument('--n-epochs', type=int, default=100000, help='the number of epochs to train the agent')
    parser.add_argument('--n-batches', type=int, default=300, help='the times to update the network')
    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--num-workers', type=int, default=1, help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=5, help='ratio to be replace')
    parser.add_argument('--batch-size', type=int, default=128, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=0.5, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.0001, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.0001, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.99, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')
    parser.add_argument('--demo-length', type=int, default=20, help='the demo length')
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    #parser.add_argument('--num-rollouts-per-mpi', type=int, default=2, help='the rollouts per mpi')
    parser.add_argument('--network', type=str, default='critic', help='the network type')
    parser.add_argument('--metric', type=str, default='MLP', help='the corresponding metric')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda device')
    parser.add_argument('--search', type=bool, default=False)
    parser.add_argument('--plan-rate', type=float, default=0)
    parser.add_argument('--lr_decay_actor', type=int, default=100000)
    parser.add_argument('--lr_decay_critic', type=int, default=100000)
    parser.add_argument('--layer', type=int, default=7)
    parser.add_argument('--period', type=int, default=3)
    parser.add_argument('--distance', type=float, default=0.3)
    parser.add_argument('--resume', type=bool, default=True)
    parser.add_argument('--resume_epoch', type=int, default=1300)
    parser.add_argument('--path', type=str, default='saved_models/AntMaze-v1_May21_15-35-16')
    #args for the planner
    parser.add_argument('--fps', type=int, default=1)
    parser.add_argument('--landmark', type=int, default=700)
    parser.add_argument('--initial-sample', type=int, default=2000)
    parser.add_argument('--clip-v', type=float, default=-35.)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    args.path='saved_models/AntMaze-v1_May21_15-35-16'
    agent_env = gym.make('AntMaze-v1')
    test_env = gym.make('AntMazeTest-v1')
    env_params = get_env_params(agent_env)
    env_params['max_test_timesteps'] = test_env._max_episode_steps
    buffer = torch.load('saved_models/AntMaze-v1_May21_15-35-16/replaybuffer.pt')
    writer = SummaryWriter(log_dir='vae_runs/vae'+current_time + '_' + str(args.env_name))

    vae_net = VAE(state_dim=2, latent_dim=128)
    for i in range(0, 5300, 100):
        agent = ddpg_agent(args, agent_env, env_params, test_env, vae_net, resume=True, resume_epoch_actor=i, resume_epoch_critic=i)
        obs = test_env.reset()
        success_rate = agent._eval_test_agent()
        writer.add_scalar('data/train_vae', success_rate, i*200)
        print(i, agent._eval_test_agent())




if __name__ == '__main__':
    main()