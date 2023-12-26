import torch
import os
import sys
sys.path.append('../')

from datetime import datetime
import numpy as np
from algos.networks import *
from algos.replay_buffer import replay_buffer
from algos.her import her_sampler
import random
from tensorboardX import SummaryWriter
from planner.goal_plan import *
import torch
from goal_env.recorder import play


class dqn_agent:
    def __init__(self, args, env, env_params, test_env, resume=False, resume_epoch=0):
        self.args = args
        self.device = args.device
        self.env = env
        self.test_env = test_env
        self.env_params = env_params
        self.action_n = env.action_space.n

        self.resume = resume
        self.resume_epoch = resume_epoch
        self.init_qnets()
        if self.resume == True:
            self.Q_network.load_state_dict(
                torch.load(self.args.path + '/q_model_' + str(self.resume_epoch) + '.pt')[0])
            self.targetQ_network.load_state_dict(
                torch.load(self.args.path + '/q_model_' + str(self.resume_epoch) + '.pt')[0])

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.writer = SummaryWriter(log_dir='runs/dqn' + current_time + '_mc' + str(args.gamma) + '_' + str(
            args.plan_rate) + '_' + str(args.fps))
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
            # path to save the model
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name + "_" + current_time)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.eps = args.eps
        # load the weights into the target networks
        self.targetQ_network.load_state_dict(self.Q_network.state_dict())
        # create the optimizer
        self.q_optim = torch.optim.Adam(self.Q_network.parameters(), lr=self.args.lr)
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.args.distance)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        if args.fps == 1:
            self.planner_policy = Planner(agent=self, framebuffer=self.buffer, fps=True, \
                                          clip_v=args.clip_v, n_landmark=args.landmark,
                                          initial_sample=args.initial_sample, model=True)
        else:
            self.planner_policy = Planner(agent=self, framebuffer=self.buffer, fps=False, \
                                          clip_v=args.clip_v, n_landmark=args.landmark,
                                          initial_sample=args.initial_sample, model=True)

    def init_qnets(self):
        self.Q_network = QNetWrapper(self.env_params, self.args).to(self.device)
        self.targetQ_network = QNetWrapper(self.env_params, self.args).to(self.device)


    def learn(self):
        for epoch in range(self.args.n_epochs):
            mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
            ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
            observation = self.env.reset()
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
            for t in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    act_obs, act_g = self._preproc_inputs(obs, g)
                    action = self.explore_policy(act_obs, act_g)
                    # feed the actions into the environment
                observation_new, _, _, info = self.env.step(action.squeeze(0))
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                # append rollouts
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                ep_g.append(g.copy())
                ep_actions.append(action.copy())
                # re-assign the observation
                obs = obs_new
                ag = ag_new
            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())
            mb_obs = np.array([ep_obs])
            mb_ag = np.array([ep_ag])
            mb_g = np.array([ep_g])
            mb_actions = np.array([ep_actions])
            self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
            for n_batch in range(self.args.n_batches):
                self._update_network()
                if n_batch % self.args.period == 0:
                    self._soft_update_target_network(self.targetQ_network, self.Q_network)
            # start to do the evaluation
            if epoch % 10 == 0:
                success_rate = self._eval_agent()
                test_success_rate = self._eval_test_agent()
                # self.store_figure(epoch)
                print('[{}] epoch is: {}, eval success rate is: {:.3f}, {:.3f}'.format(datetime.now(), epoch,
                                                                                       success_rate, test_success_rate))
                torch.save([self.Q_network.state_dict()], \
                           self.model_path + '/q_model_' + str(epoch) + '.pt')
                torch.save(self.buffer, self.model_path + '/buffer.pt')
                self.writer.add_scalar('data/train', success_rate, epoch)
                self.writer.add_scalar('data/test', test_success_rate, epoch)

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        g = torch.tensor(g, dtype=torch.float32).unsqueeze(0).to(self.device)
        return obs, g

    def explore_policy(self, obs, goal):
        q_value= self.Q_network(obs, goal)
        best_actions = q_value.max(1)[1].cpu().numpy()
        if random.random() < self.eps:
            best_actions[0] = np.random.randint(self.action_n)
        return best_actions

    def test_policy(self, obs, goal):
        q_value = self.Q_network(obs, goal)
        best_actions = q_value.max(1)[1].cpu().numpy()
        return best_actions
    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = o, g
        transitions['obs_next'], transitions['g_next'] = o_next, g
        # start to do the update
        obs_cur = transitions['obs']
        g_cur = transitions['g']
        # inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next = transitions['obs_next']
        g_next = transitions['g_next']
        ag_next = transitions['ag_next']
        # inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        obs_cur = torch.tensor(obs_cur, dtype=torch.float32).to(self.device)
        g_cur = torch.tensor(g_cur, dtype=torch.float32).to(self.device)
        obs_next = torch.tensor(obs_next, dtype=torch.float32).to(self.device)
        g_next = torch.tensor(g_next, dtype=torch.float32).to(self.device)
        ag_next = torch.tensor(ag_next, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.long).to(self.device)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32).to(self.device)
        # calculate the target Q value function
        real_q_value = self.Q_network(obs_cur, g_cur).gather(1, actions_tensor)
        with torch.no_grad():
            # do the normalization
            # tricks here
            next_q_values = self.targetQ_network(obs_next, g_next)
            #print(next_q_values.max(1)[1])
            target_action = next_q_values.max(1)[1].unsqueeze(1)
            next_q_value = next_q_values.gather(1, target_action)
            next_q_value = next_q_value.detach()
            target_q_value = r_tensor + self.args.gamma * next_q_value
            target_q_value = target_q_value.detach()
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0.)

        real_q_value = self.Q_network(obs_cur, g_cur).gather(1, actions_tensor)
        td_loss = (real_q_value - target_q_value).pow(2).mean()
        forward_loss = (self.Q_network(obs_cur, ag_next).gather(1, actions_tensor)).pow(2).mean()
        td_loss += 0.5*forward_loss
        self.q_optim.zero_grad()
        td_loss.backward()
        self.q_optim.step()

    '''
    def _eval_agent(self, policy=None):
        if policy is None:
            policy = self.planner_policy
            self.planner_policy.reset()
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            # print('___start___')
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    act_obs, act_g = self._preproc_inputs(obs, g)
                    actions = self.test_policy(act_obs, act_g)
                observation_new, _, _, info = self.env.step(actions.squeeze(0))
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                # print("obs, action", obs[:2], actions)
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
            # print(per_success_rate, total_success_rate)
        total_success_rate = np.array(total_success_rate)
        global_success_rate = np.mean(total_success_rate[:, -1])
        return global_success_rate
        
    def _eval_test_agent(self, policy=None):
        total_success_rate = []
        if policy is None:
            policy = self.planner_policy
            self.planner_policy.reset()
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            self.planner_policy.reset()
            observation = self.test_env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for num in range(self.env_params['max_test_timesteps']):
                with torch.no_grad():
                    act_obs, act_g = self._preproc_inputs(obs, g)
                    # print(act_obs.shape, act_g.shape, act_obs, act_g)
                    actions = policy(act_obs, act_g)
                observation_new, rew, done, info = self.test_env.step(actions.squeeze(0))
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
            # print(per_success_rate, total_success_rate)
        total_success_rate = np.array(total_success_rate)
        global_success_rate = np.mean(total_success_rate[:, -1])
        return global_success_rate
    '''

    def pairwise_value(self, obs, goal):
        assert obs.shape[0] == goal.shape[0]
        q_values = self.Q_network.base(obs, goal)
        best_action = q_values.min(1)[1].unsqueeze(1)
        dist = q_values.gather(1, best_action).squeeze(-1)
        #print(dist, dist.shape)
        return -dist