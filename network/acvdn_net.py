import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class ACVDNNet(nn.Module):
    def __init__(self, input_shape, args):
        super(ACVDNNet, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.leakyrelu = nn.LeakyReLU(self.args.alpha)
        if args.two_hyper_layers:
            self.emb = nn.Sequential(nn.Linear(args.state_shape, args.acvdn_hidden_dim1),
                                          nn.ReLU(),
                                          nn.Linear(args.acvdn_hidden_dim1, args.acvdn_hidden_dim2))
            self.a = nn.Parameter(torch.empty(size=(self.args.head_num, self.head_num, 2 * args.acvdn_hidden_dim, 1)))
            nn.init.xavier_uniform_(self.a.data, gain=1.414)
        else:
            self.emb = nn.Linear(input_shape, args.acvdn_hidden_dim)
            self.a = nn.Parameter(torch.empty(size=(self.args.head_num, 2* args.acvdn_hidden_dim, 1)))
            nn.init.xavier_uniform_(self.a.data, gain=1.414)


    def adj_build(self, actions):
        # action size [epsiode, max_episode_len, n_agent, 1]
        episode_num = actions.size(0)
        max_episode_len = actions.size(1)
        n_agent = actions.size(2)
        actions = actions.view(episode_num, max_episode_len, n_agent)
        actions = actions.view(-1, n_agent)# actions: [episode_num * max_episode_len, n_agents]
        channel_action_up = (np.array(actions) % (self.args.n_channel * self.args.n_channel)) // self.args.n_channel
        channel_action_down = (np.array(actions) % (self.args.n_channel * self.args.n_channel)) % self.args.n_channel
        uplink_graph = np.zeros([episode_num * max_episode_len, n_agent, n_agent])
        downlink_graph = np.zeros([episode_num * max_episode_len, n_agent, n_agent])
        for i in range(uplink_graph.shape[0]):
            for j in range(uplink_graph.shape[1]):
                idx_up = np.where(channel_action_up[i, j] == channel_action_up[i, :])
                idx_down = np.where(channel_action_down[i, j] == channel_action_down[i, :])
                uplink_graph[i, j, idx_up] = 1
                downlink_graph[i, j, idx_down] = 1
        uplink_graph = uplink_graph.reshape([episode_num, max_episode_len, n_agent, n_agent])
        downlink_graph = downlink_graph.reshape([episode_num, max_episode_len, n_agent, n_agent])
        uplink_graph = torch.from_numpy(uplink_graph)
        downlink_graph = torch.from_numpy(downlink_graph)
        att_up_count = torch.sum(uplink_graph, dim=-1)
        att_down_count = torch.sum(downlink_graph, dim=-1)
        return uplink_graph, downlink_graph, att_up_count, att_down_count

    def attention_vec_input(self, w_h, head_idx):
        # w_h: [episode_num, max_episode_len, n_agents, args.acvdn_hidden_dim]
        # self.a: [2 * args.acvdn_hidden_dim, 1]
        w_h1 = torch.matmul(w_h, self.a[head_idx, :self.args.acvdn_hidden_dim, :]) # w_h1: [episode_num, max_episode_len, n_agents, 1]
        w_h2 = torch.matmul(w_h, self.a[head_idx, self.args.acvdn_hidden_dim:, :]) # w_h2: [episode_num, max_episode_len, n_agents, 1]
        w_h2 = torch.transpose(w_h2, 2, 3)
        # broadcast add
        e = w_h1 + w_h2
        e = self.leakyrelu(e)
        return e

    def forward(self, q_values, states, obs, actions):
        # states: [episode_num, max_episode_len， state_shape]
        # q_values: [episode_num, max_episode_len， n_agents]
        # obs: [episode_num, max_episode_len, n_agents, obs_shape]
        # actions: [episode_num, max_episode_len, n_agents, 1]
        episode_num = q_values.size(0)
        adj_up, adj_down, att_up_count, att_down_count = self.adj_build(actions)
        w_h = self.emb(obs) # w_h: [episode_num, max_episode_len, n_agents, args.acvdn_hidden_dim]
        multi_head_attn_up, multi_head_attn_dn = [], []
        for head_idx in range(self.args.head_num):
            e = self.attention_vec_input(w_h, head_idx) # e: [episode_num, max_episode_len, n_agents, n_agents]
            zero_vec = 0 * torch.ones_like(e)
            att_up = torch.where(adj_up > 0.1, e, zero_vec)
            att_up = F.softmax(att_up, dim=-1)
            # att_up = F.dropout(att_up, self.dropout, training=self.training) # att_up: [episode_num, max_episode_len, n_agents, n_agents]
            att_up = att_up.view(-1, self.args.n_agents,
                                 self.args.n_agents)  # att_up: [episode_num * max_episode_len, n_agents, n_agents]

            att_down = torch.where(adj_down > 0, e, zero_vec)
            att_down = F.softmax(att_down, dim=-1)
            # att_down = F.dropout(att_down, self.dropout, training=self.training) # att_down: [episode_num, max_episode_len, n_agents, n_agents]
            att_down = att_down.view(-1, self.args.n_agents,
                                     self.args.n_agents)  # att_down: [episode_num * max_episode_len, n_agents, n_agents]

            multi_head_attn_up.append(att_up)
            multi_head_attn_dn.append(att_down)

        multi_head_attn_up = torch.stack(multi_head_attn_up, dim=0)  # multi_head_attn_up: [head_num, episode_num * max_episode_len, n_agents, n_agents]
        multi_head_attn_dn = torch.stack(multi_head_attn_dn, dim=0)  # multi_head_attn_dn: [head_num, episode_num * max_episode_len, n_agents, n_agents]
        final_attn_up = torch.mean(multi_head_attn_up, dim=0)  # multi_head_attn_up: [episode_num * max_episode_len, n_agents, n_agents]
        final_attn_dn = torch.mean(multi_head_attn_dn, dim=0)  # multi_head_attn_dn: [episode_num * max_episode_len, n_agents, n_agents]

        q_values = q_values.view(-1, 1, self.args.n_agents)  # q_values: [episode_num * max_episode_len, 1, n_agents]
        if self.args.chnl_num_attn:
            att_calibrated_q_up = torch.bmm(q_values, final_attn_up)  # [episode_num * max_episode_len, 1, n_agents]
            att_calibrated_q_down = torch.bmm(q_values, final_attn_dn)  # [episode_num * max_episode_len, 1, n_agents]
            channel_attn_up = (F.softmax(att_up_count.view([-1, self.args.n_agents]), dim=-1)).view(-1, 1, self.args.n_agents)
            channel_attn_down = (F.softmax(att_down_count.view([-1, self.args.n_agents]), dim=-1)).view(-1, 1, self.args.n_agents)
            att_calibrated_q_up = channel_attn_up.mul(att_calibrated_q_up)
            att_calibrated_q_down = channel_attn_down.mul(att_calibrated_q_down)
        else:
            att_calibrated_q_up = torch.bmm(q_values, final_attn_up)  # [episode_num * max_episode_len, 1, n_agents]
            att_calibrated_q_down = torch.bmm(q_values, final_attn_dn)  # [episode_num * max_episode_len, 1, n_agents]
        q_total = att_calibrated_q_up + att_calibrated_q_down # means of uplink and downlink Q-values
        q_total = torch.sum(q_total, dim=2, keepdim=True)
        q_total = q_total.view(episode_num, -1, 1)
        return q_total
