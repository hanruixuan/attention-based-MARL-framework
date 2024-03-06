import torch
import os
from network.base_net import MLP, D3QN
from network.vdn_net import VDNNet

class VDN:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape
        if args.reuse_network:
            input_shape += self.n_agents
        self.args = args

        self.eval_mlp = D3QN(input_shape, args)  # 每个agent选动作的网络
        self.target_mlp = D3QN(input_shape, args)
        self.eval_vdn_net = VDNNet()  # 把agentsQ值加起来的网络
        self.target_vdn_net = VDNNet()
        self.args = args
        if self.args.cuda:
            self.eval_mlp.cuda()
            self.target_mlp.cuda()
            self.eval_vdn_net.cuda()
            self.target_vdn_net.cuda()

        self.model_dir = args.model_dir + '/' + args.alg + '/'
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/mlp_net_params.pkl'):
                path_mlp = self.model_dir + '/mlp_net_params.pkl'
                path_vdn = self.model_dir + '/vdn_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_mlp.load_state_dict(torch.load(path_mlp, map_location=map_location))
                self.eval_vdn_net.load_state_dict(torch.load(path_vdn, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_mlp, path_vdn))
            else:
                raise Exception("No model!")

        self.target_mlp.load_state_dict(self.eval_mlp.state_dict())
        self.target_vdn_net.load_state_dict(self.eval_vdn_net.state_dict())

        self.eval_parameters = list(self.eval_vdn_net.parameters()) + list(self.eval_mlp.parameters())
        self.optimizer = torch.optim.Adam(self.eval_parameters, lr=args.lr, eps=args.adam_eps)
        print('Init alg VDN')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        u, r, avail_u, avail_u_next, terminated = batch['u'], batch['r'],  batch['avail_u'], \
                                                  batch['avail_u_next'], batch['terminated']
        mask = 1 - batch["padded"].float()  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
        if self.args.cuda:
            u = u.cuda()
            r = r.cuda()
            mask = mask.cuda()
            terminated = terminated.cuda()

        q_evals, q_targets, next_actions = self.get_q_values(batch, max_episode_len)
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        q_targets = torch.gather(q_targets, dim=3, index=next_actions).squeeze(3)

        q_total_eval = self.eval_vdn_net(q_evals)
        q_total_target = self.target_vdn_net(q_targets)

        targets = r + self.args.gamma * q_total_target * (1 - terminated)
        td_error = targets.detach() - q_total_eval
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_mlp.load_state_dict(self.eval_mlp.state_dict())
            self.target_vdn_net.load_state_dict(self.eval_vdn_net.state_dict())

    def _get_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)

        # 给obs添加上一个动作、agent编号
        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets, next_actions = [], [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
            q_eval = self.eval_mlp(inputs)  # 得到的q_eval维度为(episode_num*n_agents, n_actions)
            q_target = self.target_mlp(inputs_next)
            q_eval_next = self.eval_mlp(inputs_next)
            next_action = torch.max(q_eval_next, -1, keepdim=True)[1]

            # 把q_eval维度重新变回(episode_num, n_agents, n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            next_action = next_action.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
            next_actions.append(next_action)
        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        next_actions = torch.stack(next_actions, dim=1)
        return q_evals, q_targets, next_actions

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_vdn_net.state_dict(), self.model_dir + '/' + num + '_vdn_net_params.pkl')
        torch.save(self.eval_mlp.state_dict(),  self.model_dir + '/' + num + '_mlp_net_params.pkl')