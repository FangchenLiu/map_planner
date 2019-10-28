import argparse

"""
Here are the param for the training

"""

def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env-name', type=str, default='mcar-v0', help='the environment name')
    parser.add_argument('--test', type=str, default='mcar-v1')
    parser.add_argument('--n-epochs', type=int, default=3000, help='the number of epochs to train the agent')
    parser.add_argument('--n-batches', type=int, default=150, help='the times to update the network')
    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--num-workers', type=int, default=1, help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--lr', type=float, default=0.0003, help='the learning rate of the network')
    parser.add_argument('--polyak', type=float, default=0.995, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')
    parser.add_argument('--demo-length', type=int, default=20, help='the demo length')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda device')
    parser.add_argument('--period', type=int, default=1)
    parser.add_argument('--distance', type=float, default=0.08)
    parser.add_argument('--eps', type=float, default=0.2, help='epsilon greedy')
    parser.add_argument('--plan-rate', type=float, default=0, help='plan rate')
    #args for the planner
    parser.add_argument('--resume',type=bool, default=False)
    parser.add_argument('--resume-epoch', type=int, default=0)
    parser.add_argument('--path', type=str, default='saved_models/')
    parser.add_argument('--fps', type=int, default=1)
    parser.add_argument('--random-epoch', type=int, default=0)
    parser.add_argument('--landmark', type=int, default=15)
    parser.add_argument('--initial-sample', type=int, default=200)
    parser.add_argument('--clip-v', type=float, default=-8)
    parser.add_argument('--heat', type=float, default=0.8)
    args = parser.parse_args()
    return args
