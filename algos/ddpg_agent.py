import torch
import os
import sys
sys.path.append('../')
from datetime import datetime
from tensorboardX import SummaryWriter
import numpy as np
from algos.networks import *
from algos.replay_buffer import replay_buffer
from algos.her import her_sampler
from goal_env.recorder import play
from matplotlib import pyplot as plt
from planner.sample import *
from planner.vis_pointmaze import vis_pointmaze
from planner.goal_plan import *

class ddpg_agent:
    def __init__(self, args, env, env_params, test_env):
        self.args = args
        self.env = env
        self.test_env = test_env
        self.env_params = env_params
        self.device = args.device
        self.resume = args.resume
        self.resume_epoch = args.resume_epoch
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.writer = SummaryWriter(log_dir='runs/ddpg'+current_time + '_' + str(args.env_name) + \
                                            str(args.lr_critic)+'_' + str(args.gamma)+'_'+\
                                            str(args.fps))
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
            # path to save the model
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name+"_"+current_time)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.actor_network = actor(env_params)
        self.actor_target_network = actor(env_params)
        self.critic_network = criticWrapper(self.env_params, self.args)
        self.critic_target_network = criticWrapper(self.env_params, self.args)

        self.start_epoch = 0
        if self.resume == True:
            self.start_epoch = self.resume_epoch
            self.actor_network.load_state_dict(torch.load(self.args.resume_path + \
                                                          '/actor_model_' +str(self.resume_epoch) +'.pt')[0])
            self.critic_network.load_state_dict(torch.load(self.args.resume_path + \
                                                           '/critic_model_' +str(self.resume_epoch) +'.pt')[0])
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # if use gpu
        self.actor_network.to(self.device)
        self.critic_network.to(self.device)
        self.actor_target_network.to(self.device)
        self.critic_target_network.to(self.device)
        # create the optimizer

        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.args.distance)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        self.planner_policy = Planner(agent=self, replay_buffer=self.buffer, fps=args.fps, \
                                          clip_v=args.clip_v, n_landmark=args.landmark, initial_sample=args.initial_sample)

    def adjust_lr_actor(self, epoch):
        lr_actor = self.args.lr_actor * (0.5 ** (epoch // self.args.lr_decay_actor))
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = lr_actor

    def adjust_lr_critic(self, epoch):
        lr_critic = self.args.lr_critic * (0.5 ** (epoch // self.args.lr_decay_critic))
        for param_group in self.critic_optim.param_groups:
            param_group['lr'] = lr_critic

    def learn(self):
        for epoch in range(self.start_epoch, self.args.n_epochs):
            if epoch > 2000 and epoch % self.args.lr_decay_actor == 0:
                self.adjust_lr_actor(epoch)
            if epoch > 2000 and epoch % self.args.lr_decay_critic == 0:
                self.adjust_lr_critic(epoch)

            ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
            observation = self.env.reset()
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
                    # start to collect samples
            for t in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    act_obs, act_g = self._preproc_inputs(obs, g)
                    action = self.explore_policy(act_obs, act_g)
                    # feed the actions into the environment
                observation_new, _, _, info = self.env.step(action)
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

            # convert them into arrays
            mb_obs = np.array([ep_obs])
            mb_ag = np.array([ep_ag])
            mb_g = np.array([ep_g])
            mb_actions = np.array([ep_actions])
            self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
            #self.store_figure(epoch)
            for n_batch in range(self.args.n_batches):
                self._update_network()
                if n_batch % self.args.period == 0:
                    self._soft_update_target_network(self.actor_target_network, self.actor_network)
                    self._soft_update_target_network(self.critic_target_network, self.critic_network)
            # start to do the evaluation
            if epoch % 10 == 0:
                success_rate = self._eval_agent()
                test_sucess_rate = self._eval_test_agent()
                #self.store_figure(epoch)
                print('[{}] epoch is: {}, eval success rate is: {:.3f}, {:.3f}'.format(datetime.now(), epoch, success_rate, test_sucess_rate))
                torch.save([self.critic_network.state_dict()], \
                           self.model_path + '/critic_model_' +str(epoch) +'.pt')
                torch.save([self.actor_network.state_dict()], \
                            self.model_path + '/actor_model_' +str(epoch) +'.pt')
                torch.save(self.buffer, self.model_path + '/replaybuffer.pt')
                self.writer.add_scalar('data/train' + self.args.env_name + self.args.metric, success_rate, epoch)
                self.writer.add_scalar('data/test' + self.args.env_name +  self.args.metric, test_sucess_rate, epoch)

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        g = torch.tensor(g, dtype=torch.float32).unsqueeze(0).to(self.device)
        return obs, g

    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        if np.random.randn() < 0.2:
            action = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                           size=self.env_params['action'])
        return action

    def explore_policy(self, obs, goal):
        pi = self.actor_network(obs, goal)
        action = self._select_actions(pi)
        return action

    def random_policy(self, obs, goal):
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                           size=self.env_params['action'])
        return random_actions

    def test_policy(self, obs, goal):
        pi = self.actor_network(obs, goal)
        # convert the actions
        actions = pi.detach().cpu().numpy().squeeze()
        return actions

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
        ag_next = transitions['ag_next']

        # start to do the update
        obs_cur = transitions['obs']
        g_cur = transitions['g']
        #inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next = transitions['obs_next']
        g_next = transitions['g_next']
        #inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        obs_cur = torch.tensor(obs_cur, dtype=torch.float32).to(self.device)
        g_cur = torch.tensor(g_cur, dtype=torch.float32).to(self.device)
        obs_next = torch.tensor(obs_next, dtype=torch.float32).to(self.device)
        g_next = torch.tensor(g_next, dtype=torch.float32).to(self.device)
        ag_next = torch.tensor(ag_next, dtype=torch.float32).to(self.device)

        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32).to(self.device)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32).to(self.device)
        # calculate the target Q value function
        with torch.no_grad():
            actions_next = self.actor_target_network(obs_next, g_next)
            q_next_value = self.critic_target_network(obs_next, g_next, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.critic_target_network.gamma * q_next_value
            target_q_value = target_q_value.detach()
            clip_return = 1 / (1 - self.critic_target_network.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0.)
        # the q loss
        real_q_value = self.critic_network(obs_cur, g_cur, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        forward_loss = self.critic_network(obs_cur, ag_next, actions_tensor).pow(2).mean()
        critic_loss += forward_loss
        # the actor loss
        actions_real = self.actor_network(obs_cur, g_cur)
        actor_loss = -self.critic_network(obs_cur, g_cur, actions_real).mean()
        #actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 1)
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), 1)
        self.critic_optim.step()

    # do the evaluation
    def _eval_agent(self, policy=None):
        if policy is None:
            policy = self.test_policy

        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            #print('___start___')
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    act_obs, act_g = self._preproc_inputs(obs, g)
                    actions = policy(act_obs, act_g)
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                #print("obs, action", obs[:2], actions)
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
            #print(per_success_rate, total_success_rate)
        total_success_rate = np.array(total_success_rate)
        global_success_rate = np.mean(total_success_rate[:, -1])
        return global_success_rate

    def _eval_test_agent(self, policy=None):
        if policy is None:
            policy = self.planner_policy
            policy.reset()

        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.test_env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for num in range(self.env_params['max_test_timesteps']):
                with torch.no_grad():
                    act_obs, act_g = self._preproc_inputs(obs, g)
                    #print(act_obs.shape, act_g.shape, act_obs, act_g)
                    actions = policy(act_obs, act_g)
                observation_new, rew, done, info = self.test_env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
            #print(per_success_rate, total_success_rate)
        total_success_rate = np.array(total_success_rate)
        global_success_rate = np.mean(total_success_rate[:, -1])
        return global_success_rate

    def pairwise_value(self, obs, goal):
        assert obs.shape[0] == goal.shape[0]
        actions = self.actor_network(obs, goal)
        dist = self.critic_network.base(obs, goal, actions).squeeze(-1)
        return -dist

    def store_figure(self, epoch):
        dist_np = vis_pointmaze(self)
        dist_np = torch.tensor(dist_np, dtype=torch.float32)
        dist_np = dist_np.permute(2, 0, 1)
        #add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
        self.writer.add_image('image/pointmaze' + self.args.env_name + self.args.metric, dist_np, \
                              global_step=epoch)
