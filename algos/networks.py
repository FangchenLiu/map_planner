import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')
from algos.distance import *
import numpy as np

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.
"""
def initialize_metrics(metric, dim):
    if metric == 'L1':
        return L1()
    elif metric == 'L2':
        return L2()
    elif metric == 'dot':
        return DotProd()
    elif metric == 'MLP':
        return MLPDist(dim)
    else:
        raise NotImplementedError

# define the actor network
class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 400)
        self.fc4 = nn.Linear(400, 400)
        self.action_out = nn.Linear(400, env_params['action'])

    def forward(self, obs, goal):
        x = torch.cat([obs, goal], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        actions = self.max_action * torch.tanh(self.action_out(x))
        return actions

class critic(nn.Module):
    def __init__(self, env_params, args):
        super(critic, self).__init__()
        self.max_action = env_params['action_max']
        self.inp_dim = env_params['obs'] + env_params['action'] + env_params['goal']
        self.out_dim = 1
        self.mid_dim = 400

        if args.layer == 1:
            models = [nn.Linear(self.inp_dim, self.out_dim)]
        else:
            models = [nn.Linear(self.inp_dim, self.mid_dim)]
        if args.layer > 2:
            for i in range(args.layer - 2):
                models += [nn.ReLU(), nn.Linear(self.mid_dim, self.mid_dim)]
        if args.layer > 1:
            models += [nn.ReLU(), nn.Linear(self.mid_dim, self.out_dim)]

        self.base = nn.Sequential(*models)

    def forward(self, obs, goal, actions):
        x = torch.cat([obs, actions / self.max_action], dim=1)
        x = torch.cat([x, goal], dim=1)
        dist = self.base(x)
        return dist

class criticWrapper(nn.Module):
    def __init__(self, env_params, args):
        super(criticWrapper, self).__init__()
        self.base = critic(env_params, args)
        self.args = args
        #self.scale = nn.Parameter(torch.Tensor((1.,)))
        #self.gamma = nn.Parameter(torch.Tensor((args.gamma,)))
        if args.search == True:
            self.gamma = nn.Parameter(torch.Tensor((args.gamma,)))
        else: #fix gamma
            self.gamma = args.gamma
    def forward(self, obs, goal, actions):
        dist = self.base(obs, goal, actions)
        if self.args.search == True:
            self.alpha = torch.log(self.gamma)
        else:
            self.alpha = np.log(self.gamma)
        return -(1-torch.exp(dist * self.alpha))/(1-self.gamma)


class EmbedNet(nn.Module):
    def __init__(self, env_params, args):
        super(EmbedNet, self).__init__()
        self.max_action = env_params['action_max']
        self.obs_dim = env_params['obs'] + env_params['action']
        self.goal_dim = env_params['goal']
        self.out_dim = 128
        self.mid_dim = 400

        if args.layer == 1:
            obs_models = [nn.Linear(self.obs_dim, self.out_dim)]
            goal_models = [nn.Linear(self.goal_dim, self.out_dim)]
        else:
            obs_models = [nn.Linear(self.obs_dim, self.mid_dim)]
            goal_models = [nn.Linear(self.goal_dim, self.mid_dim)]
        if args.layer > 2:
            for i in range(args.layer - 2):
                obs_models += [nn.ReLU(), nn.Linear(self.mid_dim, self.mid_dim)]
                goal_models += [nn.ReLU(), nn.Linear(self.mid_dim, self.mid_dim)]
        if args.layer > 1:
            obs_models += [nn.ReLU(), nn.Linear(self.mid_dim, self.out_dim)]
            goal_models += [nn.ReLU(), nn.Linear(self.mid_dim, self.out_dim)]

        self.obs_encoder = nn.Sequential(*obs_models)
        self.goal_encoder = nn.Sequential(*goal_models)
        self.metric = initialize_metrics(args.metric, self.out_dim)

    def forward(self, obs, goal, actions):
        s = torch.cat([obs, actions / self.max_action], dim=1)
        s = self.obs_encoder(s)
        g = self.goal_encoder(goal)
        dist = self.metric(s, g)
        return dist


class StandardModel(nn.Module):
    def __init__(self, env_params, hidden_size=400):
        super(StandardModel, self).__init__()
        self.action_n = env_params['action_dim']
        self.linear1 = nn.Linear(env_params['obs']+env_params['goal'], hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, self.action_n)

    def forward(self, state, goal):
        x = torch.cat([state, goal], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class Qnet(nn.Module):
    def __init__(self, env_params):
        super(Qnet, self).__init__()
        self.mid_dim = 32
        self.metric = 'MLP'

        self.action_n = env_params['action_dim']
        self.obs_fc1 = nn.Linear(env_params['obs'], 256)
        self.obs_fc2 = nn.Linear(256, self.mid_dim * self.action_n)

        self.goal_fc1 = nn.Linear(env_params['goal'], 256)
        self.goal_fc2 = nn.Linear(256, self.mid_dim)
        if self.metric =='MLP':
            self.mlp = nn.Sequential(
                nn.Linear(self.mid_dim * (self.action_n+1), 128),
                nn.ReLU(),
                nn.Linear(128, self.action_n),
            )

    def forward(self, obs, goal):
        s = F.relu(self.obs_fc1(obs))
        s = F.relu(self.obs_fc2(s))
        s = s.view(s.size(0), self.action_n, self.mid_dim)

        g = F.relu(self.goal_fc1(goal))
        g = F.relu(self.goal_fc2(g))

        if self.metric == 'L1':
            dist = torch.abs(s-g[:, None, :]).sum(dim=2)
        elif self.metric == 'dot':
            dist = -(s * g[:, None, :]).sum(dim=2)
        elif self.metric == 'L2':
            dist = ((torch.abs(s - g[:, None, :]) ** 2).sum(dim=2) + 1e-14) ** 0.5
        elif self.metric == 'MLP':
            s = s.view(s.size(0), -1)
            x = torch.cat([s, g], dim=1)
            dist = self.mlp(x)
        else:
            raise NotImplementedError
        return dist


class QNetWrapper(nn.Module):
    def __init__(self, env_params, args):
        super(QNetWrapper, self).__init__()
        self.base = StandardModel(env_params)
        self.args = args
        self.gamma = args.gamma

    def forward(self, obs, goal):
        dist = self.base(obs, goal)
        self.alpha = np.log(self.gamma)
        qval = -(1-torch.exp(dist * self.alpha))/(1-self.gamma)
        return qval


class EmbedNetWrapper(nn.Module):
    def __init__(self, env_params, args):
        super(EmbedNetWrapper, self).__init__()
        self.base = EmbedNet(env_params, args)
        self.args = args
        #self.scale = nn.Parameter(torch.Tensor((1.,)))
        #self.gamma = nn.Parameter(torch.Tensor((args.gamma,)))
        if args.search == True:
            self.gamma = nn.Parameter(torch.Tensor((args.gamma,)))
        else: #fix gamma
            self.gamma = args.gamma
    def forward(self, obs, goal, actions):
        dist = self.base(obs, goal, actions)
        if self.args.search == True:
            self.alpha = torch.log(self.gamma)
        else:
            self.alpha = np.log(self.gamma)
        return -(1-torch.exp(dist * self.alpha))/(1-self.gamma)

class Coskx(nn.Module):
    def __init__(self, k):
            super(Coskx, self).__init__()
            self.k = k
    def forward(self, input):
        return (input * self.k).cos()

from torch.nn import init

class RND(nn.Module):
    def __init__(self, d, ptb):
        super(RND, self).__init__()
        self.target = nn.Sequential(
            nn.Linear(d, 512),
            Coskx(ptb),
            nn.Linear(512, 512)
        )
        self.predictor = nn.Sequential(
            nn.Linear(d, 512),
            nn.Sigmoid(),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, 512)
        )
        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()
        for param in self.target.parameters():
            param.requires_grad = False

    def get_feature(self, states, action):
        x = torch.cat([states, action], dim=1)
        target_feature = self.target(x)
        predict_feature = self.predictor(x)
        return predict_feature, target_feature

    def forward(self, states, action):
        x = torch.cat([states, action], dim=1)
        target_feature = self.target(x)
        predict_feature = self.predictor(x)
        return ((predict_feature - target_feature) ** 2).mean(-1)



from torch.distributions.categorical import Categorical

class Actor2(nn.Module):
    def __init__(self, d, num_actions):
        super(Actor2, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(d, 512),
            nn.Sigmoid(),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, num_actions)
        )
        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()
    def forward(self, states):
        action_scores = self.actor(states)
        action_probs = F.softmax(action_scores, dim = 1)
        return Categorical(action_probs).sample(), action_probs

class Critic2(nn.Module):
    def __init__(self, d):
        super(Critic2, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(d, 512),
            nn.Sigmoid(),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, 1)
        )
        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()
    def forward(self, states):
        values = self.critic(states)
        return values.view(-1)


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

MAX_LOG_STD = 0.5
MIN_LOG_STD = -20

def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_size=128):
        print('init vae')
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, latent_dim)
        self.log_std = nn.Linear(hidden_size, latent_dim)
        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        mean = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=MIN_LOG_STD, max=MAX_LOG_STD)
        return mean, log_std


class Decoder(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_size=128):
        print('init vae')
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear3 = torch.nn.Linear(hidden_size, out_dim)
        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class VAE(nn.Module):
    def __init__(self, state_dim, hidden_size=128, latent_dim=128):
        print('init vae')
        super(VAE, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = Encoder(state_dim, latent_dim=latent_dim, hidden_size=self.hidden_size)
        self.decoder = Decoder(latent_dim, state_dim, hidden_size=self.hidden_size)

    def forward(self, state):
        mu, log_sigma = self.encoder(state)
        sigma = torch.exp(log_sigma)
        sample = mu + torch.randn_like(mu)*sigma
        self.z_mean = mu
        self.z_sigma = sigma

        return self.decoder(sample)

    def to(self, device):
        self.encoder.to(device)
        self.decoder.to(device)

    def get_next_states(self, states):
        mu, log_sigma = self.encoder(states)
        return self.decoder(mu)

    def get_loss(self, state, next_state):
        next_pred = self.get_next_states(state)
        return ((next_state-next_pred)**2).mean()

    def train(self, input, target, epoch, optimizer, batch_size=128, beta=0.1, label=None):
        idxs = np.arange(input.shape[0])
        np.random.shuffle(idxs)
        num_batch = int(np.ceil(idxs.shape[-1] / batch_size))
        for epoch in range(epoch):
            idxs = np.arange(input.shape[0])
            np.random.shuffle(idxs)
            for batch_num in range(num_batch):
                batch_idxs = idxs[batch_num * batch_size : (batch_num + 1) * batch_size]
                train_in = input[batch_idxs].float()
                train_targ = target[batch_idxs].float()
                optimizer.zero_grad()
                dec = self.forward(train_in)
                if label is None:
                    reconstruct_loss = ((train_targ-dec)**2).mean()
                else:
                    #for positive item, label = 0
                    #for negative item, label=delta^2
                    reconstruct_vec = ((train_targ-dec)**2).mean(-1)
                    print(reconstruct_vec.shape)
                    reconstruct_loss = ((reconstruct_vec-label)**2).mean()

                ll = latent_loss(self.z_mean, self.z_sigma)
                loss = reconstruct_loss + beta*ll
                loss.backward()
                optimizer.step()
        val_input = input[idxs]
        val_target = target[idxs]
        val_dec = self.get_next_states(val_input)
        loss = ((val_target-val_dec)**2).mean().item()
        print('vae loss', loss)
        return loss