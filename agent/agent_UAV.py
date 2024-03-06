import numpy as np
import torch
from torch.distributions import Categorical


class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        if args.alg == 'acvdn':
            from policy.acvdn_UAV import ACVDN
            self.policy = ACVDN(args)
        elif args.alg == 'vdn':
            from policy.vdn_UAV import VDN
            self.policy = VDN(args)
        elif args.alg == 'qmix':
            from policy.qmix_UAV import QMIX
            self.policy = QMIX(args)
        self.args = args

    def choose_action(self, obs, agent_num, epsilon, evaluate=False):
        inputs = obs.copy()
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()

        q_value = self.policy.eval_mlp(inputs)

        if evaluate:
            action = torch.argmax(q_value)
        else:
            if np.random.uniform() < epsilon:
                action = np.random.choice(self.n_actions)
            else:
                action = torch.argmax(q_value)
        return action

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            max_episode_len_tmp = 0
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 0:
                    max_episode_len_tmp += 1
            max_episode_len = np.maximum(max_episode_len, max_episode_len_tmp)
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)
