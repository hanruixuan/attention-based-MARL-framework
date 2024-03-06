import torch.nn as nn
import torch.nn.functional as f


class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class MLP(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(MLP, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.mlp_hidden_dim1)
        self.fc2 = nn.Linear(args.mlp_hidden_dim1, args.mlp_hidden_dim2)
        self.fc3 = nn.Linear(args.mlp_hidden_dim2, args.n_actions)

    def forward(self, obs):
        x = f.relu(self.fc1(obs))
        x = f.relu(self.fc2(x))
        q = self.fc3(x)
        return q


class D3QN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(D3QN, self).__init__()
        self.hidden_output_size = 64
        self.hidden_va_size = 32
        self.args = args
        self.hidden = nn.Sequential(nn.Linear(input_shape, 64), nn.ReLU(),
                                    nn.Linear(64, 128), nn.ReLU(),
                                    nn.Linear(128, 256), nn.ReLU(),
                                    nn.Linear(256, self.hidden_output_size), nn.ReLU(), nn.ReLU())
        self.fc_h_v = nn.Linear(self.hidden_output_size, self.hidden_va_size)
        self.fc_h_a = nn.Linear(self.hidden_output_size, self.hidden_va_size)
        self.fc_v = nn.Linear(self.hidden_va_size, 1)
        self.fc_a = nn.Linear(self.hidden_va_size, self.args.n_actions)
        print('D3QN')

    def forward(self, obs):
        x = self.hidden(obs)
        v = self.fc_v(f.relu(self.fc_h_v(x)))  # Value stream
        a = self.fc_a(f.relu(self.fc_h_a(x)))  # Advantage stream
        q = v + a - a.mean(-1, keepdim=True)  # Combine streams
        return q