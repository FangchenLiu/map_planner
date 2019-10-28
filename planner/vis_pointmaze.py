import torch
import numpy as np
import argparse
import tqdm
import sys
sys.path.append('../')
import cv2
import gym

def transform(p, maze_size_scaling):
    p = p/maze_size_scaling * 8
    return (p + 4)/24

def vis_pointmaze(agent):
    agent_env = gym.make('PointMaze-v2')

    points = [agent_env.goal_space.sample() for _ in range(500)]
    goal = agent_env.reset()['desired_goal']
    landmarks = points
    landmarks.append(goal)
    landmarks = torch.Tensor(np.array(landmarks)).to(agent.device)
    maze_size_scaling = agent_env.maze_size_scaling

    init = torch.Tensor(agent_env.reset()['observation'][landmarks.shape[1]:]).to(agent.device)

    def make_obs(goal):
        a = init[None, :].expand(len(goal), *init.shape)
        a = torch.cat((goal, a), dim=1)
        return a

    dists = []
    for i in tqdm.tqdm(landmarks):
        obs = make_obs(landmarks)
        goal=i[None, :].expand_as(landmarks)
        with torch.no_grad():
            dists.append(agent.pairwise_value(obs, goal))

    dists = torch.stack(dists, dim=1).squeeze(-1)

    goal_set = np.zeros((512, 512, 3))
    for idx, i in enumerate(transform(landmarks, maze_size_scaling) * 512):
        c = int((1 - (-dists[idx, -1])/(-dists[:, -1].min())) * 240 + 10)
        cv2.circle(goal_set, (int(i[0]), int(i[1])), 5, (c, c, c), -1)
        if idx == len(landmarks) - 1:
            cv2.circle(goal_set, (int(i[0]), int(i[1])), 8, (110, 110, 10), -1)

    return goal_set
