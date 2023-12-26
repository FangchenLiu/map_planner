import tqdm
import torch
import numpy as np
from .sample import farthest_point_sample
from torch.distributions import Categorical
from sklearn.cluster import KMeans
import cv2

def transform(p):
    p = p/4 * 8
    return (p + 4)/24

class Planner:
    def __init__(self, agent, replay_buffer, heat=0.9, n_landmark=200, initial_sample=1000, fps=False, clip_v=-4,
                 fixed_landmarks=None, test_policy=True):
        self.agent = agent
        self.explore_policy = agent.explore_policy
        self.replay_buffer = replay_buffer

        self.n_landmark = n_landmark
        self.initial_sample = initial_sample
        self.fixed_landmarks = fixed_landmarks
        self.fps = fps
        self.clip_v = clip_v
        self.heat = heat
        self.flag = None
        self.saved_goal = None
        if test_policy:
            self.policy = self.agent.test_policy
        else:
            self.policy = self.agent.explore_policy
        self.time = 0

    def clip_dist(self, dists, reserve=True):
        v = self.clip_v
        if reserve:
            mm = torch.min((dists - 1000 * torch.eye(len(dists)
                                                     ).to(dists.device)).max(dim=0)[0], dists[0]*0 + v)
            dists = dists - (dists < mm[None, :]).float() * 1000000
        else:
            dists = dists - (dists < v).float() * 1000000
        return dists

    def _value_iteration(self, A, B):
        #return (A[:, :, None] + B[None, :, :]).max(dim=1)
        A = A[:, :, None] + B[None, :, :]
        d = torch.softmax(A * self.heat, dim=1)
        return (A*d).sum(dim=1), d

    def value_iteration(self, dists):
        cc = dists * (1.-torch.eye(len(dists))).to(dists.device)
        ans = cc
        for i in range(20):
            ans = self._value_iteration(ans, ans)[0]
        to = self._value_iteration(cc, ans)[1]
        return ans, to

    def make_obs(self, init, goal):
        a = init[None, :].expand(len(goal), *init.shape)
        a = torch.cat((goal, a), dim=1)
        return a

    def pairwise_dists(self, states, landmarks):
        with torch.no_grad():
            dists = []
            for i in landmarks:
                obs = states
                goal = i[None, :].expand(len(states), *i.shape)
                dists.append(self.agent.pairwise_value(obs, goal))
        return torch.stack(dists, dim=1)

    def reset(self):
        self.saved_goal = None

    def update(self, obs, goal):
        if isinstance(goal, torch.Tensor):
            goal = goal.detach().cpu().numpy()
        if isinstance(obs, np.ndarray):
            obs = torch.Tensor(obs).to(self.agent.device)
        if self.saved_goal is not None:
            if ((self.saved_goal - goal)**2).sum() < 1e-5:
                return self.landmarks, self.dists

        self.saved_goal = goal

        if self.fixed_landmarks is None:
            landmarks = self.replay_buffer.get_all_data()['ag']
            landmarks = landmarks.reshape(-1, landmarks.shape[2])

            state = self.replay_buffer.get_all_data()['obs']
            state = state.reshape(-1, state.shape[2])


            if self.fps:
                random_idx = np.random.choice(len(landmarks), self.initial_sample)
                state = state[random_idx]
                landmarks = landmarks[random_idx]

                idx = farthest_point_sample(landmarks, self.n_landmark, device=self.agent.device)
                state = state[idx]
                landmarks = landmarks[idx]
            else:
                random_idx = np.random.choice(len(landmarks), self.n_landmark)
                state = state[random_idx]
                landmarks = landmarks[random_idx]
        else:
            landmarks = np.load('landmarks.npy')
            state = np.load('state.npy')
            #raise NotImplementedError
        #print('reset explore policy for given goal')

        state = torch.Tensor(state).to(self.agent.device)
        landmarks = torch.Tensor(landmarks).to(self.agent.device)
        gg = torch.Tensor(goal).to(self.agent.device)[None,:]
        #print(landmarks.shape, gg.shape)
        landmarks = torch.cat((landmarks,gg), dim=0)
        #print('update to', obs, goal, '  ', 'landmarks:', len(landmarks))

        self.landmarks = landmarks
        dists = self.pairwise_dists(state, landmarks)

        dists = torch.min(dists, dists*0)
        dists = torch.cat((dists, dists[-1:,:]*0-100000), dim=0)

        dists = self.clip_dist(dists)
        dists, to = self.value_iteration(dists)
        self.dists = dists[:, -1]

        self.to = to[:,-1]
        return self.landmarks, self.dists

    '''
    def cluster_clip(self, labels, dists):
        labels = torch.cat((labels, torch.tensor((self.n_cluster + 1.,)).to(self.agent.device)), dim=0)
        mat1 = labels.expand(labels.size(0), labels.size(0))
        mat2 = mat1.transpose(1, 0) - mat1
        dists = torch.where(torch.abs(mat2) > 0, dists, torch.tensor((-100000.,)).to(self.agent.device))
        return dists
    '''

    def visualize_planner(self, dists, flag):
        IMAGE_SIZE = 512
        maze_size = self.agent.env.maze_size
        goal_set = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
        for idx, i in enumerate(transform(self.landmarks) * 512):
            c = int((1 - (-dists[idx, -1])/(-dists[:, -1].min())) * 240 + 10)
        cv2.circle(goal_set, (int(i[0]), int(i[1])), 5, (c, c, c), -1)
        if idx == len(self.landmarks) - 1:
            cv2.circle(goal_set, (int(i[0]), int(i[1])), 8, (110, 110, 10), -1)
        print(dists[:, -1], dists[:,-1].min(), dists[:,-1].max())
        cv2.imwrite('goal_set'+str(flag)+'.jpg', goal_set)

    def __call__(self, obs, goal=None):
        if isinstance(obs, np.ndarray):
            obs = torch.Tensor(obs).to(self.agent.device)

        if isinstance(goal, np.ndarray):
            goal = torch.Tensor(goal).to(self.agent.device)
        #add ultimate goal to landmarks
        self.update(obs[0], goal[0])
        assert len(obs) == 1
        expand_obs = obs.expand(len(self.landmarks), *obs.shape[1:])
        landmarks = self.landmarks
        obs2ld = self.clip_dist(self.agent.pairwise_value(expand_obs, landmarks), reserve=False)
        dist = obs2ld + self.dists
        #print(dist)
        if obs2ld[-1] < -10:
            # untrusted region
            idx = Categorical(torch.softmax(dist * self.heat, dim=-1)).sample((1,))
            goal = self.landmarks[idx]
        else:
            goal = goal
        return self.policy(obs, goal)