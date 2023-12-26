import argparse

"""
Here are the param for the training

"""

def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env-name', type=str, default='Bitflip-v0', help='the environment name')
    parser.add_argument('--test', type=str, default='Bitflip-v0', help='the test environment name')
    parser.add_argument('--n-epochs', type=int, default=100, help='the number of epochs to train the agent')
    parser.add_argument('--n-batches', type=int, default=40, help='the times to update the network')
    parser.add_argument('--seed', type=int, default=123, help='random seed')

    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--future-step', type=int, default=200, help='future step to be sampled')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    parser.add_argument('--batch-size', type=int, default=128, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
    parser.add_argument('--lr', type=float, default=0.001, help='the learning rate of the network')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')

    parser.add_argument('--metric', type=str, default='MLP', help='the corresponding metric')
    parser.add_argument('--device', type=str, default='cpu', help='cuda device')

    parser.add_argument('--period', type=int, default=3, help='target update period')
    parser.add_argument('--distance', type=float, default=0.3,  help='distance threshold for HER')
    parser.add_argument('--eps', type=float, default=0.2, help='epsilon greedy')

    parser.add_argument('--resume', action='store_true', help='resume or not')
    # Will be considered only if resume is True
    parser.add_argument('--resume-epoch', type=int, default=10000, help='resume epoch')
    parser.add_argument('--resume-path', type=str, default='saved_models/AntMazeTest-v1_May23_04-14-15', help='resume path')

    #args for the planner
    parser.add_argument('--fps', action='store_true', help='if use fps in the planner')
    parser.add_argument('--landmark', type=int, default=20, help="number of landmarks")
    parser.add_argument('--initial-sample', type=int, default=200, help="number of initial candidates for landmarks")
    parser.add_argument('--clip-v', type=float, default=-8., help="clip bound for the planner")

    args = parser.parse_args()

    return args
