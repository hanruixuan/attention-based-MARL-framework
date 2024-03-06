import numpy as np
import math
import copy
import random


class Multi_UAV_env_multi_level_practical_multicheck():

    def __init__(self, vel_levels):
        self.check_num = 5
        self.current_location = np.zeros((6, 2))
        self.sinr_AV = 3    # in dB
        self.sinr_TU = 8   # in dB
        self.pw_AV = 30    # in dBm
        self.pw_BS = 40  # in dBm
        self.pw_TU = 23    # in dBm
        self.g_main = 10    # in real value
        self.g_side = 1    # in real value
        self.N0 = -120    # in dBm
        self.reference_AV = -60  # in dB
        self.height_BS = 15
        self.height_AV = 150
        self.height_TU = 1.5
        self.n_agents = 10
        self.n_a_agents = 6
        self.n_g_agents = 4
        self.n_bs = 4
        self.n_speed = vel_levels
        self.n_channel = 5
        self.n_actions = self.n_speed * self.n_channel * self.n_channel
        self.episode_limit = 50
        self.arrive_reward = 10
        self.all_arrived_reward = 2000
        self.confict_reward = -100
        self.collision_reward = -100
        self.move_reward = -1
        self.episode_step = 0
        self.v_max = 50
        self.delta_t = 2
        self.adding = 500
        self.max_dist = 1500
        self.vel_actions = self.get_vel_action()
        self.trajectory = np.array([[math.sqrt(3) / 3, 750, 0, 750, math.sqrt(3)*750, 1500],
                                    [-math.sqrt(3) / 3, 1750, 433, 1500, 1732, 750],
                                    [math.inf, 0, 1299, 1500, 1299, 0],
                                    [math.sqrt(3) / 3, -250, 1732, 750, 433, 0],
                                    [-math.sqrt(3) / 3, 750, 1299, 0, 0, 750],
                                    [math.inf, 0, 433, 0, 433, 1500]])
        self.init_location = np.array([[0, 750],
                                       [433, 1500],
                                       [1299, 1500],
                                       [1732, 750],
                                       [1299, 0],
                                       [433, 0]], dtype=float)
        self.dest_location = np.array([[math.sqrt(3)*750, 1500],
                                       [1732, 750],
                                       [1299, 0],
                                       [433, 0],
                                       [0, 750],
                                       [433, 1500]], dtype=float)
        self.BS_locations = np.array([[250, 375],
                                      [250, 1100],
                                      [1482, 1100],
                                      [1482, 375]], dtype=float)
        self.alpha = 0.01
        self.beta = 10
        self.evaluate_epsidon_num = 50
        self.g_user_locations = np.zeros([self.n_g_agents, 2])
        self.convert_dB_value()
        self.evaluation_gue_location_set = self.scen_eval_loader()
        self.train_gue_location_set = self.scen_train_loader()
        print('Init Multi-level Env')
        print(self.sinr_TU)
        print(self.sinr_AV)

    def convert_dB_value(self):
        self.sinr_AV_real = 10 ** (self.sinr_AV / 10)
        self.sinr_TU_real = 10 ** (self.sinr_TU / 10)
        self.pw_AV_real = (10 ** (self.pw_AV / 10)) / 1000
        self.pw_BS_real = (10 ** (self.pw_BS / 10)) / 1000
        self.pw_TU_real = (10 ** (self.pw_TU / 10)) / 1000
        self.N0_real = (10 ** (self.N0 / 10)) / 1000
        self.reference_AV_real = 10 ** (self.reference_AV / 10)
        self.v_dist_sqr_AV_BS = (self.height_AV - self.height_BS) ** 2
        self.v_dist_sqr_TU_BS = (self.height_TU - self.height_BS) ** 2
        return

    def get_vel_action(self):
        if self.n_speed == 2:
            vel_actions = np.array([0, self.v_max])
        if self.n_speed == 3:
            vel_actions = np.array([0, self.v_max/2, self.v_max])
        if self.n_speed == 5:
            vel_actions = np.array([0, 12.5, 25, 37.5, self.v_max])
        if self.n_speed == 6:
            vel_actions = np.array([0, 10, 20, 30, 40, self.v_max])
        if self.n_speed == 11:
            vel_actions = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, self.v_max])
        return vel_actions

    def scen_eval_loader(self):
        path = 'scenario_sim_evaluation.txt'
        f2 = open(path, "r")
        lines = f2.read()
        all_scene = np.array(lines.split(','), dtype=float)
        all_scene = all_scene.reshape([100, 4, 2])
        return all_scene

    def scen_train_loader(self):
        path = 'scenario_sim_training.txt'
        train_loc = []
        f2 = open(path, "r")
        lines = f2.readlines()
        for line in lines:
            a = line.split(', ')
            train_loc.append(a)
        train_loc = np.array(train_loc, dtype=float)
        train_loc = train_loc.reshape([-1, 4, 2])
        return train_loc

    def get_state_size(self):
        arrive_state = self.n_agents
        onehot_bs = self.n_agents * self.n_bs
        location = self.n_agents * 2
        dim = arrive_state + onehot_bs + location
        return dim

    def get_obs_size(self):
        remianing_dist = 1
        onehot_bs = (self.n_agents - 1) * self.n_bs
        safe_dist = 1 * (self.n_agents - 1)
        dist_to_bs = 1 * (self.n_agents - 1)
        dim = remianing_dist + onehot_bs + safe_dist + dist_to_bs + self.n_agents + self.n_agents
        return dim

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.n_actions,
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit,
                    "n_channel": self.n_channel}
        return env_info

    def evaluate_location_set(self):
        self.evaluation_gue_location_set = np.zeros([self.evaluate_epsidon_num, self.n_g_agents, 2])
        self.evaluation_gue_dist_set = np.zeros([self.evaluate_epsidon_num, self.n_g_agents])
        self.evaluation_gue_angle_set = np.zeros([self.evaluate_epsidon_num, self.n_g_agents])
        for i in range(self.evaluate_epsidon_num):
            for j in range(self.n_g_agents):
                g_user_dist = random.randint(50, 200)
                g_user_angle = random.uniform(0, 2 * math.pi)
                dist_x = math.cos(g_user_angle) * g_user_dist
                dist_y = math.sin(g_user_angle) * g_user_dist
                gue_location_array = np.array([dist_x, dist_y])
                self.evaluation_gue_location_set[i, j, :] = self.BS_locations[j, :] + gue_location_array
                self.evaluation_gue_dist_set[i, j] = g_user_dist
                self.evaluation_gue_angle_set[i, j] = g_user_angle

    def reset(self, evaluate, slice, evaluate_idx, train_idx):
        self.episode_step = 0
        self.gue_location_initial(evaluate, slice, evaluate_idx, train_idx)
        self.current_location = np.copy(self.init_location)
        self.current_location = np.concatenate((self.current_location, self.g_user_locations), axis=0)
        self.is_arrived = [0 for i in range(self.n_agents)]
        self.recorded_arrive = []
        return

    def gue_location_initial(self, evaluate, slice, evaluate_idx, train_idx):
        if evaluate:
            idx = 10 * slice + evaluate_idx
            self.g_user_locations = self.evaluation_gue_location_set[idx]
        else:
            self.g_user_locations = self.train_gue_location_set[train_idx]
        return

    def delta_location(self, idx, action):
        vel_level_idx = action // (self.n_channel * self.n_channel)
        vel = self.vel_actions[vel_level_idx]
        moving_dist = vel * self.delta_t
        condition = self.trajectory[idx].squeeze()
        direction = self.trajectory[idx, 5] - self.trajectory[idx, 3]
        if condition[0] >= 0 and direction.squeeze() >= 0:
            a = math.cos(math.atan(condition[0]))
            b = math.sin(math.atan(condition[0]))
            delta_x = moving_dist * a
            delta_y = moving_dist * b
        elif condition[0] >= 0 and direction < 0:
            delta_x = moving_dist * -math.cos(math.atan(condition[0]))
            delta_y = moving_dist * -math.sin(math.atan(condition[0]))
        elif condition[0] < 0 and direction >= 0:
            delta_x = moving_dist * -math.cos(math.atan(-condition[0]))  # delta_x > 0
            delta_y = moving_dist * math.sin(math.atan(-condition[0]))  # delta_y < 0
        elif condition[0] < 0 and direction < 0:
            delta_x = moving_dist * math.cos(math.atan(-condition[0]))  # delta_x < 0
            delta_y = moving_dist * -math.sin(math.atan(-condition[0]))  # delta_y > 0
        delta_dist = np.array([delta_x, delta_y])
        return delta_dist

    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_obs_agent(self, agent_id):
        feat = []
        if self.is_arrived[agent_id] == 1:
            feat.append(0)
            feat = np.array(feat)
            for al_id in range(self.n_agents):
                bs = np.zeros(self.n_bs, dtype=np.float32)
                if al_id == agent_id:
                    continue
                else:
                    dist, safe_dist, bs_idx = 0, 0, 0
                    feat = np.append(feat, bs)
                    feat = np.append(feat, dist)
                    feat = np.append(feat, safe_dist)
            # addition arrived index
            feat = np.append(feat, np.zeros(self.n_agents))
            # addition agent_id
            feat = np.append(feat, np.zeros(self.n_agents))
        else:
            if agent_id < self.n_a_agents:
                feat.append(1)
                feat.append(1)   # type symbol
                feat = np.array(feat)
                for al_id in range(self.n_agents):
                    bs = np.zeros(self.n_bs, dtype=np.float32)
                    if al_id == agent_id:
                        continue
                    else:
                        if self.is_arrived[al_id] == 0:
                            dist, safe_dist, bs_idx = self.obs_calculation(agent_id, al_id)
                            bs[bs_idx] = 1
                        else:
                            dist, safe_dist, bs_idx = 0, 0, 0
                        feat = np.append(feat, bs)
                        feat = np.append(feat, dist)
                        feat = np.append(feat, safe_dist)
                # addition arrived index
                add_arrived_idx = []
                for idx, arrive_flag in enumerate(self.is_arrived):
                    if idx == agent_id:
                        continue
                    else:
                        if arrive_flag == 1:
                            add_arrived_idx.append(0)
                        else:
                            add_arrived_idx.append(1)
                feat = np.append(feat, np.array(add_arrived_idx))
                feat_id = np.zeros(self.n_agents)
                feat_id[agent_id] = 1
                feat = np.append(feat, feat_id)
            else:
                feat.append(1)
                feat.append(0)  # type symbol
                feat = np.array(feat)
                for al_id in range(self.n_agents):
                    bs = np.zeros(self.n_bs, dtype=np.float32)
                    if al_id == agent_id:
                        continue
                    else:
                        if self.is_arrived[al_id] == 0:
                            dist, safe_dist, bs_idx = self.obs_calculation(agent_id, al_id)
                            bs[bs_idx] = 1
                        else:
                            dist, safe_dist, bs_idx = 0, 0, 0
                        feat = np.append(feat, bs)
                        feat = np.append(feat, dist)
                        feat = np.append(feat, safe_dist)
                # addition arrived index
                add_arrived_idx = []
                for idx, arrive_flag in enumerate(self.is_arrived):
                    if idx == agent_id:
                        continue
                    else:
                        if arrive_flag == 1:
                            add_arrived_idx.append(0)
                        else:
                            add_arrived_idx.append(1)
                feat = np.append(feat, np.array(add_arrived_idx))
                feat_id = np.zeros(self.n_agents)
                feat_id[agent_id] = 1
                feat = np.append(feat, feat_id)
        obs_n = feat
        return obs_n

    def obs_calculation(self, idx1, idx2):
        idx1_location = self.current_location[idx1].squeeze()
        idx2_location = self.current_location[idx2].squeeze()
        BS_location = np.array(self.BS_locations)

        dist_idx2_BS = np.sum((BS_location - idx2_location) ** 2, axis=1)
        associate_idx2_BS = np.argmin(dist_idx2_BS)
        min_dist_idx2_BS = np.min(dist_idx2_BS)
        safe_dist_idx2_BS = math.sqrt(min_dist_idx2_BS) + self.adding
        safe_dist_idx2_BS = safe_dist_idx2_BS / self.max_dist

        b = BS_location[associate_idx2_BS, :]
        a = np.sum((b - idx1_location) ** 2)
        dist_idx1_BS = math.sqrt(a) / self.max_dist

        return dist_idx1_BS, safe_dist_idx2_BS, associate_idx2_BS

    def get_state(self):
        state = np.zeros(self.get_state_size(), dtype=np.float32)
        return state

    def get_avail_agent_actions(self, agent_id):
        actions = [1 for i in range(self.n_actions)]
        return actions

    def step(self, actions):
        actions_int = [int(a) for a in actions]
        inter_locations = np.zeros([self.n_a_agents, self.check_num, 2], dtype=float)

        for a_id, action in enumerate(actions_int):
            if a_id < self.n_a_agents and self.is_arrived[a_id] == 0:
                delta_dist = self.delta_location(a_id, action)
                interval = delta_dist / (self.check_num - 1)
                for check_loc_idx in range(self.check_num):
                    inter_locations[a_id, check_loc_idx, :] = self.current_location[a_id] + check_loc_idx * interval
                self.current_location[a_id] = self.current_location[a_id] + delta_dist

        self.episode_step += 1
        reward, is_conflict_up, is_conflict_down, is_collision, rate_up, rate_down = self._reward(actions_int,
                                                                                                  inter_locations)

        is_arrived = np.copy(self.is_arrived)
        b = self.is_arrived[:self.n_a_agents]
        terminated = all(b)
        info = {"actions": actions_int,
                "arrived": is_arrived,
                "all_arrived": terminated,
                "uplink conflict": is_conflict_up,
                "downlink conflict": is_conflict_down,
                "collision": is_collision,
                "rate_up": rate_up,
                "rate_down": rate_down}

        return reward, terminated, info

    def _reward(self, action, inter_locations):
        reward = 0

        for a_id, ac in enumerate(action):
            if a_id < self.n_a_agents:
                self.is_arrived[a_id] = self.is_arrived_cal(a_id)

        b = self.is_arrived[:self.n_a_agents]
        if all(b):
            reward += self.all_arrived_reward

        # single uav arrived reward
        for a_id, arrived in enumerate(self.is_arrived):
            if a_id not in self.recorded_arrive and arrived == 1:
                reward += self.arrive_reward
                self.recorded_arrive.append(a_id)

        is_conflict_up_AV = self.check_action_conflict_AV_up(action, inter_locations)
        is_conflict_up_TU, rate_up = self.check_action_conflict_TU_up(action, inter_locations)
        for conf_flag in is_conflict_up_AV:
            if conf_flag:
                reward += self.confict_reward
        for conf_flag in is_conflict_up_TU:
            if conf_flag:
                reward += self.confict_reward

        is_conflict_down_AV = self.check_action_conflict_AV_down(action, inter_locations)
        is_conflict_down_TU, rate_down = self.check_action_conflict_TU_down(action, inter_locations)
        for conf_flag in is_conflict_down_AV:
            if conf_flag:
                reward += self.confict_reward
        for conf_flag in is_conflict_down_TU:
            if conf_flag:
                reward += self.confict_reward

        # action conflict reward
        is_collision = self.check_action_collision()
        if any(is_collision):
            reward += self.confict_reward

        # move reward
        for a_id in range(self.n_a_agents):
            if self.is_arrived[a_id] != 1:
                reward -= self.beta
        for a_id in range(self.n_g_agents):
            rate_reward = self.alpha * (rate_up[a_id] + rate_down[a_id])
            reward += rate_reward
        is_conflict_up = [any(is_conflict_down_AV), any(is_conflict_down_TU)]
        is_conflict_down = [any(is_conflict_down_AV), any(is_conflict_down_TU)]
        return reward, is_conflict_up, is_conflict_down, is_collision, rate_up, rate_down

    def check_action_conflict_AV_up(self, actions, inter_locations):
        # inter_locations: [AV, check_idx, loc]
        is_conflict_a_agent = [False for i in range(self.n_agents)]
        all_actions = (np.array(actions) % (self.n_channel * self.n_channel)) // self.n_channel
        g_actions = all_actions[self.n_a_agents:]
        a_actions = all_actions[:self.n_a_agents]
        for check_idx in range(self.check_num):
            for a_id, action in enumerate(a_actions):
                if self.is_arrived[a_id] == 1:
                    continue
                else:
                    associate_bs, horizontal_dist_sqr = self.safe_region_cal(inter_locations[a_id, check_idx])
                    dist_sqr = horizontal_dist_sqr + self.v_dist_sqr_AV_BS
                    channel_gain = self.reference_AV_real / dist_sqr
                    desired_signal = self.pw_AV_real * channel_gain  # in watt

                    # interference from TU
                    interf_from_TU = 0
                    matching_g_user = np.argwhere(g_actions == action)
                    for usr in matching_g_user:
                        dist_interf = math.sqrt(
                            np.sum((self.g_user_locations[usr[0]] - self.BS_locations[
                                associate_bs]) ** 2) + self.v_dist_sqr_TU_BS)
                        path_loss_interf = 32.4 + 20 * math.log(2, 10) + 30 * math.log(dist_interf,
                                                                                       10) + 7.8  # last term shadow fadeing
                        channel_gain_interf = 10 ** (-path_loss_interf / 10)
                        interf_signal = self.pw_TU_real * channel_gain_interf
                        interf_from_TU += interf_signal

                    # interference from AV
                    interf_from_AV = 0
                    matching_a_user = np.argwhere(a_actions == action)
                    for usr in matching_a_user:
                        if usr[0] != a_id and self.is_arrived[usr[0]] == 0:
                            dist_sqr_interf = np.sum((inter_locations[usr[0], check_idx] -
                                                      self.BS_locations[associate_bs]) ** 2) + self.v_dist_sqr_AV_BS
                            channel_gain_interf = self.reference_AV_real / dist_sqr_interf
                            interf_signal = self.pw_AV_real * channel_gain_interf
                            interf_from_AV += interf_signal

                    uplink_sinr_AV = desired_signal / (interf_from_TU + interf_from_AV + self.N0_real)
                    if uplink_sinr_AV < self.sinr_AV_real:
                        is_conflict_a_agent[a_id] = True

        return is_conflict_a_agent

    def check_action_conflict_AV_down(self, actions, inter_locations):
        is_conflict_a_agent = [False for i in range(self.n_agents)]
        all_actions = (np.array(actions) % (self.n_channel * self.n_channel)) % self.n_channel
        g_actions = all_actions[self.n_a_agents:]
        a_actions = all_actions[:self.n_a_agents]
        # inter_locations: [AV, check_idx, loc]
        for check_idx in range(self.check_num):
            for a_id, action in enumerate(a_actions):
                if self.is_arrived[a_id] == 1:
                    continue
                else:
                    associate_bs, horizontal_dist_sqr = self.safe_region_cal(inter_locations[a_id, check_idx])
                    dist_sqr = horizontal_dist_sqr + self.v_dist_sqr_AV_BS
                    channel_gain = self.g_side * self.reference_AV_real / dist_sqr
                    desired_signal = self.pw_BS_real * channel_gain

                    # interference from other TU_BS
                    interf_from_TU = 0
                    matching_user_g = np.argwhere(g_actions == action)
                    for usr in matching_user_g:
                        dist_sqr_interf = np.sum((inter_locations[a_id, check_idx] -
                                                  self.BS_locations[usr[0]]) ** 2) + self.v_dist_sqr_AV_BS
                        channel_gain_interf = self.g_side * self.reference_AV_real / dist_sqr_interf
                        interf_signal = self.pw_BS_real * channel_gain_interf
                        interf_from_TU += interf_signal

                    # interference from other AV_BS
                    interf_from_AV = 0
                    matching_user_a = np.argwhere(a_actions == action)
                    for usr in matching_user_a:
                        if usr[0] != a_id and self.is_arrived[usr[0]] == 0:
                            associate_bs_others, _ = self.safe_region_cal(inter_locations[usr[0], check_idx])
                            dist_sqr_interf = np.sum((inter_locations[a_id, check_idx] -
                                                      self.BS_locations[
                                                          associate_bs_others]) ** 2) + self.v_dist_sqr_AV_BS
                            channel_gain_interf = self.g_side * self.reference_AV_real / dist_sqr_interf
                            interf_signal = channel_gain_interf * self.pw_BS_real
                            interf_from_AV += interf_signal

                    downlink_sinr_AV = desired_signal / (interf_from_TU + interf_from_AV + self.N0_real)
                    if downlink_sinr_AV < self.sinr_AV_real:
                        is_conflict_a_agent[a_id] = True

        return is_conflict_a_agent

    def check_action_conflict_TU_up(self, actions, inter_locations):
        all_actions = (np.array(actions) % (self.n_channel * self.n_channel)) // self.n_channel
        g_actions = all_actions[self.n_a_agents:]
        a_actions = all_actions[:self.n_a_agents]
        is_conflict_g_agent = [False for i in range(self.n_g_agents)]
        rate = [0 for i in range(self.n_g_agents)]
        for a_id, action in enumerate(g_actions):
            dist = math.sqrt(
                np.sum((self.g_user_locations[a_id] - self.BS_locations[a_id]) ** 2) + self.v_dist_sqr_TU_BS)
            path_loss = 32.4 + 20 * math.log(2, 10) + 30 * math.log(dist, 10) + 7.8
            channel_gain = 10 ** (-path_loss / 10)
            desired_signal = self.pw_TU_real * channel_gain

            # interference from TU
            interf_from_TU = 0
            matching_g_user = np.argwhere(g_actions == action)
            for usr in matching_g_user:
                if usr[0] != a_id:
                    dist_interf = math.sqrt(np.sum((self.g_user_locations[usr[0]] -
                                                    self.BS_locations[a_id]) ** 2) + self.v_dist_sqr_TU_BS)
                    path_loss_interf = 32.4 + 20 * math.log(2, 10) + 30 * math.log(dist_interf, 10) + 7.8
                    channel_gain_interf = 10 ** (-path_loss_interf / 10)
                    interf_signal = self.pw_TU_real * channel_gain_interf
                    interf_from_TU += interf_signal

            # interference from AV
            uplink_sinr_TU = []
            for check_idx in range(self.check_num):
                interf_from_AV = 0
                matching_a_user = np.argwhere(action == a_actions)
                for usr in matching_a_user:
                    if self.is_arrived[usr[0]] == 0:
                        dist_sqr_interf = np.sum((inter_locations[usr[0], check_idx] -
                                                  self.BS_locations[a_id]) ** 2) + self.v_dist_sqr_AV_BS
                        channel_gain_interf = self.reference_AV_real / dist_sqr_interf
                        interf_signal = self.pw_AV_real * channel_gain_interf
                        interf_from_AV = interf_signal

                uplink_sinr_TU_temp = desired_signal / (interf_from_TU + interf_from_AV + self.N0_real)
                uplink_sinr_TU.append(uplink_sinr_TU_temp)
            uplink_sinr_TU = np.array(uplink_sinr_TU).mean()
            if uplink_sinr_TU < self.sinr_TU_real:
                is_conflict_g_agent[a_id] = True
            rate[a_id] = math.log((1 + uplink_sinr_TU), 2)
        return is_conflict_g_agent, rate

    def check_action_conflict_TU_down(self, actions, inter_locations):
        all_actions = (np.array(actions) % (self.n_channel * self.n_channel)) % self.n_channel
        g_actions = all_actions[self.n_a_agents:]
        a_actions = all_actions[:self.n_a_agents]
        is_conflict_g_agent = [False for i in range(self.n_g_agents)]
        rate = [0 for i in range(self.n_g_agents)]
        for a_id, action in enumerate(g_actions):
            dist = math.sqrt(np.sum((self.g_user_locations[a_id] -
                                     self.BS_locations[a_id]) ** 2) + self.v_dist_sqr_TU_BS)
            path_loss = 32.4 + 20 * math.log(2, 10) + 30 * math.log(dist, 10) + 7.8
            channel_gain = self.g_main * 10 ** (-path_loss / 10)
            desired_signal = self.pw_BS_real * channel_gain

            # interference from other TU_BS
            interf_from_TU = 0
            matching_user_g = np.argwhere(g_actions == action)
            for usr in matching_user_g:
                if usr[0] != a_id:
                    dist_interf = math.sqrt(np.sum((self.g_user_locations[a_id] -
                                                    self.BS_locations[usr[0]]) ** 2) + self.v_dist_sqr_TU_BS)
                    path_loss_interf = 32.4 + 20 * math.log(2, 10) + 30 * math.log(dist_interf, 10) + 7.8
                    channel_gain_interf = self.g_main * 10 ** (-path_loss_interf / 10)
                    interf_signal = self.pw_BS_real * channel_gain_interf
                    interf_from_TU += interf_signal

            # interference from other AV_BS
            downlink_sinr_TU = []
            for check_idx in range(self.check_num):
                interf_from_AV = 0
                matching_user_a = np.argwhere(action == a_actions)
                for usr in matching_user_a:
                    if self.is_arrived[usr[0]] == 0:
                        associate_bs_others, _ = self.safe_region_cal(inter_locations[usr[0], check_idx])
                        dist_interf = math.sqrt(np.sum((self.g_user_locations[a_id] - self.BS_locations[
                            associate_bs_others]) ** 2) + self.v_dist_sqr_TU_BS)
                        path_loss_interf = 32.4 + 20 * math.log(2, 10) + 30 * math.log(dist_interf, 10) + 7.8
                        channel_gain_interf = self.g_main * 10 ** (-path_loss_interf / 10)
                        interf_signal = self.pw_BS_real * channel_gain_interf
                        interf_from_AV += interf_signal

                downlink_sinr_TU_tmp = desired_signal / (interf_from_TU + interf_from_AV + self.N0_real)
                downlink_sinr_TU.append(downlink_sinr_TU_tmp)
            downlink_sinr_TU = np.array(downlink_sinr_TU).mean()
            if downlink_sinr_TU < self.sinr_TU_real:
                is_conflict_g_agent[a_id] = True
            rate[a_id] = math.log((1 + downlink_sinr_TU), 2)
        return is_conflict_g_agent, rate

    def safe_region_cal(self, loc):
        dist_BS_all = np.sum((self.BS_locations - loc) ** 2, axis=1)
        associate_BS = np.argmin(dist_BS_all)
        dist_bs = np.min(dist_BS_all)
        return associate_BS, dist_bs

    def is_arrived_cal(self, idx):
        condition = self.trajectory[idx].squeeze()
        direction = self.trajectory[idx, 5] - self.trajectory[idx, 3]
        arrived = 0
        if condition[0] >= 0 and direction >= 0:
            if self.current_location[idx][1] - self.trajectory[idx, 5] >= -1:
                arrived = 1
        elif condition[0] >= 0 and direction < 0:
            if self.current_location[idx][1] - self.trajectory[idx, 5] <= 5:
                arrived = 1
        elif condition[0] < 0 and direction >= 0:
            if self.current_location[idx][1] - self.trajectory[idx, 5] >= -1:
                arrived = 1
        elif condition[0] < 0 and direction < 0:
            if self.current_location[idx][1] - self.trajectory[idx, 5] <= 5:
                arrived = 1
        return arrived

    def check_action_collision(self):
        is_collision = [False for i in range(self.n_a_agents)]
        for i in range(self.n_a_agents):
            if self.is_arrived[i] == 1:
                continue
            else:
                dist_all = np.sqrt(np.sum((self.current_location[:self.n_a_agents, :] - self.current_location[i]) ** 2, axis=1))
                dist_all[self.is_arrived == 0] = 10000
                dist_all[dist_all == 0] = 1
                dist_all[i] = 0
                dist_all[dist_all > 100] = 0
                if np.sum(dist_all) > 0:
                    is_collision[i] = True
        return is_collision