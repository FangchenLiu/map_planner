import gym
import numpy as np
import sys
sys.path.append('../')
from algos.random_policy import RandomPolicy
from goal_env import *
from goal_env.recorder import play

env0 = gym.make('GoalPlane-v1')
p = RandomPolicy(env0.action_space)
play(env0, p, 'tmp.avi', 200)