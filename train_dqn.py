import numpy as np
import gym
import os
import sys
from arguments_dqn import get_args
from mpi4py import MPI
from subprocess import CalledProcessError
from algos.dqn_agent import dqn_agent
from goal_env import *
import random
import torch
from tensorboardX import SummaryWriter
writer = SummaryWriter()

def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': 1,
            'action_max': 1,
            'action_dim': env.action_space.n
            }
    params['max_timesteps'] = env._max_episode_steps
    print(params)
    return params

def launch(args):
    env = gym.make(args.env_name)
    # set random seeds for reproduce
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device is not 'cpu':
        torch.cuda.manual_seed(args.seed)
    # get the environment parameters
    env_params = get_env_params(env)
    from algos.dqn_agent import dqn_agent
    dqn_agent = dqn_agent(args, env, env_params, writer)
    dqn_agent.learn()

if __name__ == '__main__':
    # get the params
    args = get_args()
    launch(args)
