import numpy as np
import os


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args
        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init RolloutWorker')

    def generate_episode(self, train_idx=None, slice=None, episode_num=None, evaluate=False, evaluate_steps=None):
        o, u, r, s, u_onehot, terminate, padded = [], [], [], [], [], [], []
        info_tot = []
        self.env.reset(evaluate, slice, episode_num, train_idx)
        terminated = False
        win_tag = False
        collision_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while not terminated and step < self.episode_limit:
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []
            for agent_id in range(self.n_agents):
                action = self.agents.choose_action(obs[agent_id], agent_id, epsilon, evaluate)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append(action_onehot)

            reward, terminated, info = self.env.step(actions)
            win_tag = True if terminated and 'is_arrived' in info and info['is_arrived'] else False
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
            if evaluate:
                info_tot.append(info)
        # last obs
        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
        if evaluate:
            self.write_log(info_tot, episode_num, evaluate_steps+1)
            win_tag = False
            collision_tag = False
            for info in info_tot:
                win_tag = any(info['uplink conflict']) or any(info['downlink conflict']) or win_tag
            for info in info_tot:
                collision_tag = any(info['collision']) or collision_tag
        return episode, episode_reward, win_tag, step, collision_tag

    def write_log(self, info_tot, episode_num, evaluate_steps):
        save_path = './log/' + self.args.alg + '_results/' + str(evaluate_steps) + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = save_path + str(episode_num) + '.txt'
        with open(file_path, 'w') as f:
            location_string = 'GUE_locations: ' + str(list(self.env.g_user_locations.reshape([-1]))) + '\n'
            f.write(location_string)
            for idx, info in enumerate(info_tot):
                string = 'step ' + str(idx) + '\n'
                string = string + 'actions: ' + str(info['actions']) + '\n'
                string = string + 'arrived: ' + str(info['arrived']) + '\n'
                string = string + 'all_arrived: ' + str(info['all_arrived']) + '\n'
                string = string + 'uplink conflict: ' + str(info['uplink conflict']) + '\n'
                string = string + 'downlink conflict: ' + str(info['downlink conflict']) + '\n'
                string = string + 'collision: ' + str(info['collision']) + '\n'
                string = string + 'rate_up: ' + str(info['rate_up']) + '\n'
                string = string + 'rate_down: ' + str(info['rate_down']) + '\n'
                string = string + '\n'
                f.write(string)