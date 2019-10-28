import argparse

"""
Here are the param for the training

"""

def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env-name', type=str, default='PointMaze-v1', help='the environment name')
    parser.add_argument('--test', type=str, default='PointMazeTest-v1')
    parser.add_argument('--agent', type=str, default='td3')
    parser.add_argument('--n-epochs', type=int, default=7000, help='the number of epochs to train the agent')
    parser.add_argument('--n-batches', type=int, default=120, help='the times to update the network')
    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--num-workers', type=int, default=1, help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--noise-clip', type=float, default=0.5, help='noise eps')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--update-policy', type=int, default=2)
    parser.add_argument('--replay-k', type=int, default=5, help='ratio to be replace')
    parser.add_argument('--batch-size', type=int, default=128, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.975, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.0003, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.0001, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.98, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')
    parser.add_argument('--demo-length', type=int, default=20, help='the demo length')
    #parser.add_argument('--num-rollouts-per-mpi', type=int, default=2, help='the rollouts per mpi')
    parser.add_argument('--network', type=str, default='critic', help='the network type')
    parser.add_argument('--metric', type=str, default='MLP', help='the corresponding metric')
    parser.add_argument('--device', type=str, default='cuda:1', help='cuda device')
    parser.add_argument('--search', type=bool, default=False)
    parser.add_argument('--plan-rate', type=float, default=0.)
    parser.add_argument('--lr_decay_actor', type=int, default=1000)
    parser.add_argument('--lr_decay_critic', type=int, default=2000)
    parser.add_argument('--layer', type=int, default=7)
    parser.add_argument('--period', type=int, default=10)
    parser.add_argument('--fps', type=int, default=1)
    args = parser.parse_args()
    return args

