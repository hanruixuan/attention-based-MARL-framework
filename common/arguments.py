import argparse

"""
Here are the param for the training

"""


def get_common_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')
    parser.add_argument('--alg', type=str, default='acvdn', help='the algorithm to train the agent')
    parser.add_argument('--n_steps', type=int, default=50000000, help='total time steps')
    parser.add_argument('--n_episodes', type=int, default=1, help='the number of episodes before once training')
    parser.add_argument('--last_action', type=bool, default=False, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--evaluate_cycle', type=int, default=5000, help='how often to evaluate the model')
    parser.add_argument('--evaluate_epoch', type=int, default=1, help='number of the epoch to evaluate the agent')
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--evaluate', type=bool, default=False, help='whether to evaluate the model')
    parser.add_argument('--cuda', type=bool, default=False, help='whether to use the GPU')
    args = parser.parse_args()
    return args


# arguments of vnd, qmix
def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.mlp_hidden_dim1 = 128
    args.mlp_hidden_dim2 = 256
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.lr = 5e-4
    print(args.lr)
    args.alpha = 0.2
    args.head_num = 4
    args.chnl_num_attn = False
    args.acvdn_hidden_dim = 128
    args.acvdn_hidden_dim1 = 64
    args.acvdn_hidden_dim2 = 128

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.1
    anneal_steps = 50000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the train steps in one epoch
    args.train_steps = 1

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(2e3)

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # env velocity levels
    args.velocity_levels = 6

    # prevent gradient explosion
    args.grad_norm_clip = 10
    return args




